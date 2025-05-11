"""KV-cache-aware routing."""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from infermesh.registry import WorkerRegistry
    from infermesh.settings import GatewayConfig, WorkerConfig


class NoWorkersAvailable(Exception):
    def __init__(self, model: str) -> None:
        super().__init__(f"no healthy workers for model '{model}'")
        self.model = model


# ---------------------------------------------------------------------------
# Radix trie
# ---------------------------------------------------------------------------

@dataclass
class _TrieNode:
    children: dict[str, "_TrieNode"] = field(default_factory=dict)
    worker_id: str | None = None
    inserted_at: float = 0.0
    ttl_override_s: float | None = None


class RadixTrie:
    """Character-level radix trie with TTL eviction. Per-model roots.

    longest_prefix_match traverses as deep as the query allows and returns
    the deepest non-expired terminal worker_id seen.
    """

    def __init__(self, ttl_s: float) -> None:
        self._ttl_s = ttl_s
        self._roots: dict[str, _TrieNode] = {}

    def insert(
        self,
        model: str,
        prefix: str,
        worker_id: str,
        ttl_override_s: float | None = None,
    ) -> None:
        if model not in self._roots:
            self._roots[model] = _TrieNode()
        node = self._roots[model]
        remaining = prefix
        now = time.monotonic()

        while remaining:
            matched_edge: str | None = None
            for edge in node.children:
                if _common_prefix(edge, remaining):
                    matched_edge = edge
                    break

            if matched_edge is None:
                node.children[remaining] = _TrieNode(
                    worker_id=worker_id,
                    inserted_at=now,
                    ttl_override_s=ttl_override_s,
                )
                return

            common_len = len(_common_prefix(matched_edge, remaining))
            if common_len == len(matched_edge):
                node = node.children[matched_edge]
                remaining = remaining[common_len:]
                if not remaining:
                    node.worker_id = worker_id
                    node.inserted_at = now
                    node.ttl_override_s = ttl_override_s
            else:
                common_str = matched_edge[:common_len]
                old_suffix = matched_edge[common_len:]
                new_suffix = remaining[common_len:]

                old_child = node.children.pop(matched_edge)
                split_node = _TrieNode()
                split_node.children[old_suffix] = old_child
                node.children[common_str] = split_node

                if new_suffix:
                    split_node.children[new_suffix] = _TrieNode(
                        worker_id=worker_id,
                        inserted_at=now,
                        ttl_override_s=ttl_override_s,
                    )
                else:
                    split_node.worker_id = worker_id
                    split_node.inserted_at = now
                    split_node.ttl_override_s = ttl_override_s
                return

    def longest_prefix_match(self, model: str, prefix: str) -> tuple[str | None, int]:
        """Return (worker_id, match_length). (None, 0) when no match exists."""
        root = self._roots.get(model)
        if root is None:
            return None, 0

        node = root
        remaining = prefix
        best_worker: str | None = None
        best_depth = 0
        consumed = 0
        now = time.monotonic()

        while remaining:
            matched_edge: str | None = None
            for edge in node.children:
                if remaining.startswith(edge):
                    matched_edge = edge
                    break
                common = _common_prefix(edge, remaining)
                if common:
                    partial_depth = consumed + len(common)
                    child = node.children[edge]
                    if (
                        child.worker_id
                        and not _is_expired(child, now, self._ttl_s)
                        and partial_depth > best_depth
                    ):
                        best_worker = child.worker_id
                        best_depth = partial_depth
                    return best_worker, best_depth

            if matched_edge is None:
                break

            consumed += len(matched_edge)
            node = node.children[matched_edge]
            remaining = remaining[len(matched_edge):]

            if node.worker_id and not _is_expired(node, now, self._ttl_s):
                if consumed > best_depth:
                    best_worker = node.worker_id
                    best_depth = consumed

        return best_worker, best_depth

    def entry_count(self) -> int:
        now = time.monotonic()
        return sum(_count_live(r, now, self._ttl_s) for r in self._roots.values())

    def evict_expired(self) -> int:
        now = time.monotonic()
        removed = 0
        for model in list(self._roots.keys()):
            n, root = _evict(self._roots[model], now, self._ttl_s)
            removed += n
            if root is None:
                del self._roots[model]
        return removed

    def remove(self, model: str, prefix: str) -> bool:
        """Exact-match removal. Returns True if found."""
        root = self._roots.get(model)
        if root is None:
            return False

        node = root
        remaining = prefix
        while remaining:
            matched_edge: str | None = None
            for edge in node.children:
                if remaining.startswith(edge):
                    matched_edge = edge
                    break
            if matched_edge is None:
                return False
            node = node.children[matched_edge]
            remaining = remaining[len(matched_edge):]

        if node.worker_id is None:
            return False
        node.worker_id = None
        node.inserted_at = 0.0
        node.ttl_override_s = None
        return True

    @staticmethod
    def extract_prefix(messages: list[dict]) -> str:
        return "\n".join(
            f"{m.get('role', '')}: {m.get('content', '') or ''}" for m in messages
        )


def _common_prefix(a: str, b: str) -> str:
    i = 0
    while i < len(a) and i < len(b) and a[i] == b[i]:
        i += 1
    return a[:i]


def _is_expired(node: _TrieNode, now: float, ttl_s: float) -> bool:
    effective_ttl = node.ttl_override_s if node.ttl_override_s is not None else ttl_s
    return node.inserted_at > 0 and (now - node.inserted_at) > effective_ttl


def _count_live(node: _TrieNode, now: float, ttl_s: float) -> int:
    count = 1 if node.worker_id and not _is_expired(node, now, ttl_s) else 0
    for child in node.children.values():
        count += _count_live(child, now, ttl_s)
    return count


def _evict(node: _TrieNode, now: float, ttl_s: float) -> tuple[int, _TrieNode | None]:
    removed = 0
    if node.worker_id and _is_expired(node, now, ttl_s):
        node.worker_id = None
        node.inserted_at = 0.0
        removed += 1

    for edge in list(node.children.keys()):
        n, child = _evict(node.children[edge], now, ttl_s)
        removed += n
        if child is None:
            del node.children[edge]
        else:
            node.children[edge] = child

    if not node.children and not node.worker_id:
        return removed, None
    return removed, node


# ---------------------------------------------------------------------------
# Token estimator
# ---------------------------------------------------------------------------

def _estimate_tokens(body: dict) -> int:
    """Rough char-based token estimate. Used for ETIF, not billing."""
    messages = body.get("messages", [])
    chars = sum(len(str(m.get("content", "") or "")) for m in messages)
    prompt_est = max(1, chars // 4)
    completion_est = body.get("max_tokens") or 256
    return prompt_est + int(completion_est)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

def _kv_aware_pick(
    trie: RadixTrie,
    model: str,
    prefix: str,
    pool: list["WorkerConfig"],
    min_match: int = 0,
) -> "WorkerConfig | None":
    """Longest-prefix lookup restricted to a worker pool."""
    worker_id, depth = trie.longest_prefix_match(model, prefix)
    if worker_id and depth >= max(1, min_match):
        return next((w for w in pool if w.id == worker_id), None)
    return None


class PowerOfTwoChoicesRouter:
    """Pick two random healthy workers; return the one with lower load score.

    Score: etif_weight * (tokens_in_flight / etif_scale)
           + (1 - etif_weight) * cache_utilization
    Both terms clamped to [0, 1]. OpenAI-type workers use ETIF only.
    """

    def __init__(
        self,
        registry: "WorkerRegistry",
        etif_weight: float = 0.6,
        etif_scale: int = 4096,
    ) -> None:
        self._registry = registry
        self._etif_weight = etif_weight
        self._etif_scale = etif_scale

    def _score(self, worker: "WorkerConfig") -> float:
        etif_norm = min(worker.tokens_in_flight / self._etif_scale, 1.0)
        if worker.type == "openai":
            return etif_norm
        return (
            self._etif_weight * etif_norm
            + (1.0 - self._etif_weight) * worker.cache_utilization
        )

    def select(self, model: str) -> "WorkerConfig":
        workers = self._registry.get_workers(model)
        if not workers:
            raise NoWorkersAvailable(model)
        if len(workers) == 1:
            return workers[0]
        weights = [w.weight for w in workers]
        a = random.choices(workers, weights=weights, k=1)[0]
        rest = [w for w in workers if w is not a]
        b = random.choices(rest, weights=[w.weight for w in rest], k=1)[0]
        return a if self._score(a) <= self._score(b) else b


class KVAwareRouter:
    """Trie-first routing; falls back to PowerOfTwoChoicesRouter.

    select() returns (worker, cache_hit).
    """

    def __init__(
        self,
        registry: "WorkerRegistry",
        trie: RadixTrie,
        config: "GatewayConfig",
    ) -> None:
        self._registry = registry
        self._trie = trie
        self._p2c = PowerOfTwoChoicesRouter(
            registry,
            etif_weight=config.etif_weight,
            etif_scale=config.etif_scale,
        )
        self._config = config

    def select(
        self,
        model: str,
        messages: list[dict],
        prefix: str | None = None,
    ) -> tuple["WorkerConfig", bool]:
        if prefix is None:
            prefix = RadixTrie.extract_prefix(messages)
        pool = self._registry.get_workers(model)
        matched = _kv_aware_pick(self._trie, model, prefix, pool, self._config.kv_match_min_chars)
        if matched is not None:
            return matched, True
        return self._p2c.select(model), False


# ---------------------------------------------------------------------------
# Disaggregated prefill/decode routing
# ---------------------------------------------------------------------------

@dataclass
class _SessionEntry:
    decode_worker_id: str
    model: str
    last_seen: float


class SessionTracker:
    """Active decode sessions, keyed by session_id.

    Sliding-window TTL: a successful lookup refreshes last_seen, so active
    conversations persist while abandoned ones expire.

    asyncio-safety: every method below runs to completion without yielding.
    Dict mutations are atomic at the bytecode level. Do not introduce
    `await` inside these methods.
    """

    def __init__(self, ttl_s: float = 600.0) -> None:
        self._ttl_s = ttl_s
        self._sessions: dict[str, _SessionEntry] = {}
        self.prefill_complete_received_total = 0
        self.sticky_hits_total = 0
        self.sticky_misses_total = 0
        self.session_eviction_total = 0

    def register(self, session_id: str, decode_worker_id: str, model: str) -> None:
        self._sessions[session_id] = _SessionEntry(
            decode_worker_id=decode_worker_id,
            model=model,
            last_seen=time.monotonic(),
        )
        self.prefill_complete_received_total += 1

    def lookup(self, session_id: str) -> tuple[str, str] | None:
        entry = self._sessions.get(session_id)
        if entry is None:
            self.sticky_misses_total += 1
            return None
        now = time.monotonic()
        if now - entry.last_seen > self._ttl_s:
            del self._sessions[session_id]
            self.session_eviction_total += 1
            self.sticky_misses_total += 1
            return None
        entry.last_seen = now
        self.sticky_hits_total += 1
        return entry.decode_worker_id, entry.model

    def remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def evict_expired(self) -> int:
        now = time.monotonic()
        expired = [
            s for s, e in self._sessions.items()
            if now - e.last_seen > self._ttl_s
        ]
        for s in expired:
            del self._sessions[s]
        self.session_eviction_total += len(expired)
        return len(expired)

    def session_count(self) -> int:
        return len(self._sessions)


class DisaggregatedRouter:
    """Routes prefill and decode phases to separate pools.

    select_prefill: KV-aware longest-prefix match within the prefill pool,
        falling back to P2C(ETIF + util) on miss.
    select_decode:  sticky-route to the decode worker holding the KV for
        this session, falling back to P2C within the decode pool.

    Instantiated only when GatewayConfig.disaggregation_enabled is True.
    """

    def __init__(
        self,
        registry: "WorkerRegistry",
        trie: RadixTrie,
        sessions: SessionTracker,
        config: "GatewayConfig",
    ) -> None:
        self._registry = registry
        self._trie = trie
        self._sessions = sessions
        self._config = config
        self._p2c = PowerOfTwoChoicesRouter(
            registry,
            etif_weight=config.etif_weight,
            etif_scale=config.etif_scale,
        )

    def select_prefill(
        self,
        model: str,
        messages: list[dict],
    ) -> tuple["WorkerConfig", bool]:
        pool = self._registry.get_prefill_workers(model)
        if not pool:
            raise NoWorkersAvailable(model)
        prefix = RadixTrie.extract_prefix(messages)
        matched = _kv_aware_pick(self._trie, model, prefix, pool, self._config.kv_match_min_chars)
        if matched is not None:
            return matched, True
        return self._p2c_within(pool), False

    def select_decode(
        self,
        model: str,
        session_id: str | None,
        messages: list[dict],
    ) -> tuple["WorkerConfig", bool]:
        if session_id:
            entry = self._sessions.lookup(session_id)
            if entry is not None:
                decode_wid, sess_model = entry
                if sess_model == model:
                    pool = self._registry.get_decode_workers(model)
                    hit = next((w for w in pool if w.id == decode_wid), None)
                    if hit is not None:
                        return hit, True

        pool = self._registry.get_decode_workers(model)
        if not pool:
            raise NoWorkersAvailable(model)
        return self._p2c_within(pool), False

    def _p2c_within(self, workers: list["WorkerConfig"]) -> "WorkerConfig":
        if len(workers) == 1:
            return workers[0]
        a, b = random.sample(workers, 2)
        return a if self._p2c._score(a) <= self._p2c._score(b) else b
