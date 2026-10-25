"""KV-cache-aware routing.

Follows SGLang's cache-aware router (sgl-model-gateway/src/policies/tree.rs and
cache_aware.rs): a character-level radix tree in which every node records the
workers that have sent that prefix, a match-ratio threshold, and a load
imbalance override that skips the cache when one worker is running far ahead.
"""
from __future__ import annotations

import random
import time
from collections.abc import Callable, Collection
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
    # worker_id -> monotonic expiry. Stamped on every node an insert passes
    # through, so a shared prefix keeps its workers after the tree branches.
    tenants: dict[str, float] = field(default_factory=dict)


class RadixTrie:
    """Character-level radix trie with per-worker TTL. Per-model roots."""

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
        if not prefix:
            return
        ttl = ttl_override_s if ttl_override_s is not None else self._ttl_s
        expires = time.monotonic() + ttl
        node = self._roots.setdefault(model, _TrieNode())
        remaining = prefix

        while remaining:
            edge = _edge_for(node, remaining[0])
            if edge is None:
                node.children[remaining] = _TrieNode(tenants={worker_id: expires})
                return
            common = _common_len(edge, remaining)
            child = node.children[edge]
            if common < len(edge):
                # Split. Every tenant of the old edge holds the shared part.
                split = _TrieNode(tenants=dict(child.tenants))
                split.children[edge[common:]] = child
                del node.children[edge]
                node.children[edge[:common]] = split
                child = split
            child.tenants[worker_id] = max(child.tenants.get(worker_id, 0.0), expires)
            node = child
            remaining = remaining[common:]

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
            edge = _edge_for(node, remaining[0])
            if edge is None:
                break
            common = _common_len(edge, remaining)
            child = node.children[edge]
            live = [
                w for w, exp in child.tenants.items()
                if exp > now and (candidates is None or w in candidates)
            ]
            if not live:
                break
            best, best_depth = live, consumed + common
            if common < len(edge):
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
            edge = _edge_for(node, remaining[0])
            if edge is None:
                return False
            if remaining.startswith(edge):
                node = node.children[edge]
                remaining = remaining[len(edge):]
            elif edge.startswith(remaining):
                node = node.children[edge]
                remaining = ""
            else:
                return False
        return _drop_tenant(node, worker_id)

    def entry_count(self) -> int:
        """Nodes with at least one live worker."""
        now = time.monotonic()
        return sum(_count_live(r, now) for r in self._roots.values())

    def evict_expired(self) -> int:
        """Remove expired (node, worker) attributions; prune empty nodes."""
        now = time.monotonic()
        removed = 0
        for model in list(self._roots):
            root = self._roots[model]
            removed += _evict(root, now)
            if not root.children:
                del self._roots[model]
        return removed

    @staticmethod
    def extract_prefix(messages: list[dict]) -> str:
        return "\n".join(
            f"{m.get('role', '')}: {m.get('content', '') or ''}" for m in messages
        )


def _edge_for(node: _TrieNode, first_char: str) -> str | None:
    # Radix invariant: at most one child edge starts with a given character.
    return next((e for e in node.children if e[0] == first_char), None)


def _common_len(a: str, b: str) -> int:
    i = 0
    n = min(len(a), len(b))
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _drop_tenant(node: _TrieNode, worker_id: str) -> bool:
    found = node.tenants.pop(worker_id, None) is not None
    for child in node.children.values():
        found = _drop_tenant(child, worker_id) or found
    return found


def _count_live(node: _TrieNode, now: float) -> int:
    count = 1 if any(exp > now for exp in node.tenants.values()) else 0
    for child in node.children.values():
        count += _count_live(child, now)
    return count


def _evict(node: _TrieNode, now: float) -> int:
    expired = [w for w, exp in node.tenants.items() if exp <= now]
    for w in expired:
        del node.tenants[w]
    removed = len(expired)
    for edge in list(node.children):
        child = node.children[edge]
        removed += _evict(child, now)
        if not child.tenants and not child.children:
            del node.children[edge]
    return removed


# ---------------------------------------------------------------------------
# Token estimator
# ---------------------------------------------------------------------------

def _estimate_tokens(body: dict) -> int:
    """Rough char-based token estimate. Used for ETIF, not billing."""
    messages = body.get("messages", [])
    chars = sum(len(str(m.get("content", "") or "")) for m in messages)
    prompt_est = max(1, chars // 4)
    completion_est = body.get("max_completion_tokens") or body.get("max_tokens") or 256
    return prompt_est + int(completion_est)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

def _cache_aware_pick(
    trie: RadixTrie,
    model: str,
    prefix: str,
    pool: list["WorkerConfig"],
    config: "GatewayConfig",
) -> "WorkerConfig | None":
    """SGLang cache_aware rule. Returns None when the caller should load-balance.

    1. If load is imbalanced (both the absolute and the relative gap exceed
       their thresholds), ignore the cache.
    2. Otherwise route to a worker holding the deepest matched prefix, but
       only if the match covers more than `cache_threshold` of the prompt.
       Among several such workers take the least loaded (Ray Serve's
       PrefixCacheAffinityRouter does the same).
    """
    loads = [w.requests_in_flight for w in pool]
    lo, hi = min(loads), max(loads)
    if hi - lo > config.balance_abs_threshold and hi > lo * config.balance_rel_threshold:
        return None
    by_id = {w.id: w for w in pool}
    tenants, depth = trie.longest_prefix_match(model, prefix, by_id)
    if not tenants or depth < config.kv_match_min_chars:
        return None
    if depth <= config.cache_threshold * len(prefix):
        return None
    return min((by_id[t] for t in tenants), key=lambda w: w.requests_in_flight)


class PowerOfTwoChoicesRouter:
    """Pick two random workers (by weight); return the one with lower load score.

    Score: etif_weight * (tokens_in_flight / etif_scale)
           + (1 - etif_weight) * cache_utilization
    Both terms clamped to [0, 1]. OpenAI-type workers use ETIF only.
    """

    def __init__(self, etif_weight: float = 0.6, etif_scale: int = 4096) -> None:
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

    def pick(self, workers: list["WorkerConfig"]) -> "WorkerConfig":
        if len(workers) == 1:
            return workers[0]
        a = random.choices(workers, weights=[w.weight for w in workers], k=1)[0]
        rest = [w for w in workers if w is not a]
        b = random.choices(rest, weights=[w.weight for w in rest], k=1)[0]
        return a if self._score(a) <= self._score(b) else b


def _routable(
    workers: list["WorkerConfig"],
    exclude: Collection[str],
    is_available: Callable[["WorkerConfig"], bool],
) -> list["WorkerConfig"]:
    return [w for w in workers if w.id not in exclude and is_available(w)]


class KVAwareRouter:
    """Cache-aware pick among healthy workers; falls back to P2C.

    select() returns (worker, cache_hit).
    """

    def __init__(
        self,
        registry: "WorkerRegistry",
        trie: RadixTrie,
        config: "GatewayConfig",
        is_available: Callable[["WorkerConfig"], bool] = lambda w: True,
    ) -> None:
        self._registry = registry
        self._trie = trie
        self._config = config
        self._is_available = is_available
        self._p2c = PowerOfTwoChoicesRouter(config.etif_weight, config.etif_scale)

    def select(
        self,
        model: str,
        messages: list[dict],
        prefix: str | None = None,
        exclude: Collection[str] = (),
    ) -> tuple["WorkerConfig", bool]:
        if prefix is None:
            prefix = RadixTrie.extract_prefix(messages)
        pool = _routable(self._registry.get_workers(model), exclude, self._is_available)
        if not pool:
            raise NoWorkersAvailable(model)
        matched = _cache_aware_pick(self._trie, model, prefix, pool, self._config)
        if matched is not None:
            return matched, True
        return self._p2c.pick(pool), False


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

    select_prefill: cache-aware pick within the prefill pool, falling back to
        P2C(ETIF + util) on miss.
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
        is_available: Callable[["WorkerConfig"], bool] = lambda w: True,
    ) -> None:
        self._registry = registry
        self._trie = trie
        self._sessions = sessions
        self._config = config
        self._is_available = is_available
        self._p2c = PowerOfTwoChoicesRouter(config.etif_weight, config.etif_scale)

    def select_prefill(
        self,
        model: str,
        messages: list[dict],
        exclude: Collection[str] = (),
    ) -> tuple["WorkerConfig", bool]:
        pool = _routable(self._registry.get_prefill_workers(model), exclude, self._is_available)
        if not pool:
            raise NoWorkersAvailable(model)
        prefix = RadixTrie.extract_prefix(messages)
        matched = _cache_aware_pick(self._trie, model, prefix, pool, self._config)
        if matched is not None:
            return matched, True
        return self._p2c.pick(pool), False

    def select_decode(
        self,
        model: str,
        session_id: str | None,
        messages: list[dict],
        exclude: Collection[str] = (),
    ) -> tuple["WorkerConfig", bool]:
        pool = _routable(self._registry.get_decode_workers(model), exclude, self._is_available)
        if session_id:
            entry = self._sessions.lookup(session_id)
            if entry is not None:
                decode_wid, sess_model = entry
                if sess_model == model:
                    hit = next((w for w in pool if w.id == decode_wid), None)
                    if hit is not None:
                        return hit, True
        if not pool:
            raise NoWorkersAvailable(model)
        return self._p2c.pick(pool), False
