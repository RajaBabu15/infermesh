"""Character-level radix trie that maps prompt prefixes to the workers that served them."""
from __future__ import annotations

import heapq
import time
from collections.abc import Collection
from dataclasses import dataclass, field


@dataclass(eq=False)
class _TrieNode:
    label: str = ""                      # edge label from the parent to this node
    parent: "_TrieNode | None" = None
    # Keyed by the first character of the child's label (radix invariant: at
    # most one child edge starts with a given character), so edge lookup is O(1).
    children: dict[str, "_TrieNode"] = field(default_factory=dict)
    # worker_id -> monotonic expiry. Stamped on every node an insert passes
    # through, so a shared prefix keeps its workers after the tree branches.
    tenants: dict[str, float] = field(default_factory=dict)
    touched: float = 0.0                 # last insert or match through this node (LRU key)


class RadixTrie:
    """Character-level radix trie with per-worker TTL and a per-worker size cap.

    Traversals are iterative, so deep prompts cannot hit the recursion limit. A
    worker over `max_chars_per_worker` loses its least-recently-used leaves first.
    """

    _CAP_HEADROOM = 0.9                  # evict down to 90 percent of the cap, not 100

    def __init__(self, ttl_s: float, max_chars_per_worker: int = 2**26) -> None:
        self._ttl_s = ttl_s
        self._max_chars = max_chars_per_worker
        self._roots: dict[str, _TrieNode] = {}
        self._chars: dict[str, int] = {}

    def _attach(self, node: _TrieNode, worker_id: str, expires: float) -> None:
        if worker_id in node.tenants:
            node.tenants[worker_id] = max(node.tenants[worker_id], expires)
            return
        node.tenants[worker_id] = expires
        self._chars[worker_id] = self._chars.get(worker_id, 0) + len(node.label)

    def _detach(self, node: _TrieNode, worker_id: str) -> bool:
        if node.tenants.pop(worker_id, None) is None:
            return False
        left = self._chars.get(worker_id, 0) - len(node.label)
        if left > 0:
            self._chars[worker_id] = left
        else:
            self._chars.pop(worker_id, None)
        return True

    @staticmethod
    def _prune_up(node: _TrieNode) -> None:
        """Delete `node` and any ancestors left with no tenants and no children."""
        while node.parent is not None and not node.tenants and not node.children:
            parent = node.parent
            if parent.children.get(node.label[0]) is not node:
                return                           # already pruned by an earlier call
            del parent.children[node.label[0]]
            node = parent

    def _split(self, child: _TrieNode, common: int) -> _TrieNode:
        """Split `child`'s edge after `common` characters. Every tenant of the old
        edge holds the shared part, so characters per worker are unchanged."""
        parent = child.parent
        mid = _TrieNode(label=child.label[:common], parent=parent,
                        tenants=dict(child.tenants), touched=child.touched)
        parent.children[mid.label[0]] = mid
        child.label = child.label[common:]
        child.parent = mid
        mid.children[child.label[0]] = child
        return mid

    @staticmethod
    def _walk(root: _TrieNode):
        """Every node under `root` including it, parents before children."""
        stack = [root]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(node.children.values())

    def insert(
        self,
        model: str,
        prefix: str,
        worker_id: str,
        ttl_override_s: float | None = None,
    ) -> None:
        if not prefix:
            return
        ttl = ttl_override_s if ttl_override_s is not None else self._ttl_s
        now = time.monotonic()
        expires = now + ttl
        node = self._roots.setdefault(model, _TrieNode())
        remaining = prefix

        while remaining:
            child = node.children.get(remaining[0])
            if child is None:
                leaf = _TrieNode(label=remaining, parent=node, touched=now)
                node.children[remaining[0]] = leaf
                self._attach(leaf, worker_id, expires)
                break
            common = _common_len(child.label, remaining)
            if common < len(child.label):
                child = self._split(child, common)
            self._attach(child, worker_id, expires)
            child.touched = now
            node = child
            remaining = remaining[common:]
        self._enforce_cap(worker_id)

    def longest_prefix_match(
        self,
        model: str,
        prefix: str,
        candidates: Collection[str] | None = None,
    ) -> tuple[list[str], int]:
        """Return (live workers holding the deepest matched prefix, match length).

        A match may end inside an edge. `candidates` restricts which workers
        count. Returns ([], 0) when nothing matches.
        """
        node = self._roots.get(model)
        if node is None:
            return [], 0
        now = time.monotonic()
        best: list[str] = []
        best_depth = 0
        consumed = 0
        remaining = prefix

        while remaining:
            child = node.children.get(remaining[0])
            if child is None:
                break
            common = _common_len(child.label, remaining)
            live = [
                w for w, exp in child.tenants.items()
                if exp > now and (candidates is None or w in candidates)
            ]
            if not live:
                break
            best, best_depth = live, consumed + common
            child.touched = now
            if common < len(child.label):
                break
            node = child
            consumed += common
            remaining = remaining[common:]

        return best, best_depth

    def remove(self, model: str, prefix: str, worker_id: str) -> bool:
        """Drop worker_id from `prefix` and every extension of it.

        An engine that evicted a prefix has also evicted everything after it.
        Returns True if the worker was attributed anywhere in that subtree.
        """
        node = self._roots.get(model)
        if node is None or not prefix:
            return False
        remaining = prefix
        while remaining:
            child = node.children.get(remaining[0])
            if child is None:
                return False
            if remaining.startswith(child.label):
                remaining = remaining[len(child.label):]
            elif child.label.startswith(remaining):
                remaining = ""
            else:
                return False
            node = child
        return self._drop_worker(node, worker_id) > 0

    def remove_worker(self, worker_id: str) -> int:
        """Forget everything a worker was believed to hold (it restarted, flushed
        its cache, or the event stream lost entries). Returns attributions dropped."""
        dropped = 0
        for model in list(self._roots):
            root = self._roots[model]
            dropped += self._drop_worker(root, worker_id)
            if not root.children:
                del self._roots[model]
        return dropped

    def _drop_worker(self, top: _TrieNode, worker_id: str) -> int:
        order = list(self._walk(top))
        dropped = 0
        for node in order:
            dropped += self._detach(node, worker_id)
        for node in reversed(order):            # children before parents
            self._prune_up(node)
        return dropped

    def entry_count(self) -> int:
        """Nodes with at least one live worker."""
        now = time.monotonic()
        count = 0
        for root in self._roots.values():
            for node in self._walk(root):
                if node.parent is not None and any(exp > now for exp in node.tenants.values()):
                    count += 1
        return count

    def evict_expired(self) -> int:
        """Remove expired (node, worker) attributions; prune empty nodes."""
        now = time.monotonic()
        removed = 0
        for model in list(self._roots):
            root = self._roots[model]
            order = list(self._walk(root))
            for node in order:
                for worker_id in [w for w, exp in node.tenants.items() if exp <= now]:
                    self._detach(node, worker_id)
                    removed += 1
            for node in reversed(order):
                self._prune_up(node)
            if not root.children:
                del self._roots[model]
        return removed

    def chars_for(self, worker_id: str) -> int:
        """Characters currently attributed to `worker_id` (what the cap bounds)."""
        return self._chars.get(worker_id, 0)

    def _enforce_cap(self, worker_id: str) -> None:
        if self._max_chars <= 0 or self._chars.get(worker_id, 0) <= self._max_chars:
            return
        target = int(self._max_chars * self._CAP_HEADROOM)
        heap: list[tuple[float, int, _TrieNode]] = []

        def is_leaf(node: _TrieNode) -> bool:
            return worker_id in node.tenants and not any(
                worker_id in c.tenants for c in node.children.values())

        for root in self._roots.values():
            for node in self._walk(root):
                if node.parent is not None and is_leaf(node):
                    heapq.heappush(heap, (node.touched, id(node), node))
        while heap and self._chars.get(worker_id, 0) > target:
            _, _, node = heapq.heappop(heap)
            if not is_leaf(node):
                continue
            parent = node.parent
            self._detach(node, worker_id)
            self._prune_up(node)
            if parent is not None and parent.parent is not None and is_leaf(parent):
                heapq.heappush(heap, (parent.touched, id(parent), parent))

    @staticmethod
    def extract_prefix(messages: list[dict]) -> str:
        return "\n".join(
            f"{m.get('role', '')}: {m.get('content', '') or ''}" for m in messages
        )


def _common_len(a: str, b: str) -> int:
    """Length of the common prefix of a and b.

    Binary search over slice comparisons: the slices halve each step, so it is O(n)
    bytes compared in C across O(log n) operations, not an n-step Python loop.
    """
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n - 1                  # first mismatch index lies in [lo, hi]
    while lo < hi:
        mid = (lo + hi) // 2
        if a[lo:mid + 1] == b[lo:mid + 1]:
            lo = mid + 1
        else:
            hi = mid
    return lo
