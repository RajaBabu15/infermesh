"""KV-cache-aware routing.

Follows SGLang's cache-aware router (sgl-model-gateway/src/policies/tree.rs and
cache_aware.rs): a character-level radix tree in which every node records the
workers that have sent that prefix, a match-ratio threshold, and a load
imbalance override that skips the cache when one worker is running far ahead.

Differs from SGLang in three places: a low match falls back to power-of-two
choices, not the minimum-load worker; entries expire per worker by TTL and a
per-worker character cap, not by a periodic LRU pass over a node cap; and
worker-published prefix events (kv_subscriber) can correct the trie.
"""
from __future__ import annotations

import random
import time
from collections.abc import Callable, Collection
from typing import TYPE_CHECKING

from infermesh.trie import RadixTrie

if TYPE_CHECKING:
    from infermesh.registry import WorkerRegistry
    from infermesh.settings import GatewayConfig, WorkerConfig


class NoWorkersAvailable(Exception):
    def __init__(self, model: str) -> None:
        super().__init__(f"no healthy workers for model '{model}'")
        self.model = model


def _estimate_tokens(
    body: dict, count_output: bool = True, default_output: int = 256
) -> int:
    """Rough char-based token estimate. Used for ETIF, not billing.

    With count_output the client's max_tokens (or `default_output`) is reserved
    too, until the response ends; without it only the prompt is counted.
    """
    messages = body.get("messages", [])
    chars = sum(len(str(m.get("content", "") or "")) for m in messages)
    prompt_est = max(1, chars // 4)
    if not count_output:
        return prompt_est
    completion_est = body.get("max_completion_tokens") or body.get("max_tokens") or default_output
    return prompt_est + int(completion_est)


def _cache_aware_pick(
    trie: RadixTrie,
    model: str,
    prefix: str,
    pool: list["WorkerConfig"],
    config: "GatewayConfig",
) -> "WorkerConfig | None":
    """SGLang's cache_aware rule; None means load-balance instead.

    Ignore the cache when both the absolute and relative load gaps exceed their
    thresholds. Otherwise pick the least-loaded worker holding the deepest match,
    if the match covers more than `cache_threshold` of the prompt.
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
    """Pick two random workers (by weight) and return the one with the lower load.

    Score: etif_weight * tokens_in_flight/etif_scale + queue_weight * queue_depth/queue_scale
    + the rest * cache_utilization, each term clamped to [0, 1]. Hosted (openai-type)
    workers, and vLLM workers whose metrics are missing or stale, use ETIF only. Two
    choices instead of the global minimum, which herds onto one worker when load
    info is periodically refreshed.
    """

    def __init__(
        self,
        etif_weight: float = 0.6,
        etif_scale: int = 4096,
        queue_weight: float = 0.0,
        queue_scale: int = 8,
        max_age_s: float = float("inf"),
    ) -> None:
        self._etif_weight = etif_weight
        self._etif_scale = etif_scale
        self._queue_weight = queue_weight
        self._queue_scale = queue_scale
        self._max_age_s = max_age_s

    def _engine_metrics_fresh(self, worker: "WorkerConfig") -> bool:
        if worker.cache_utilization is None:
            return False
        at = worker.cache_utilization_at
        return at is None or time.monotonic() - at <= self._max_age_s

    def _score(self, worker: "WorkerConfig") -> float:
        etif_norm = min(worker.tokens_in_flight / self._etif_scale, 1.0)
        if worker.type == "openai" or not self._engine_metrics_fresh(worker):
            return etif_norm
        queue_norm = 0.0
        if worker.queue_depth is not None:
            queue_norm = min(worker.queue_depth / self._queue_scale, 1.0)
        util_weight = max(0.0, 1.0 - self._etif_weight - self._queue_weight)
        return (
            self._etif_weight * etif_norm
            + self._queue_weight * queue_norm
            + util_weight * worker.cache_utilization
        )

    def pick(self, workers: list["WorkerConfig"]) -> "WorkerConfig":
        if len(workers) == 1:
            return workers[0]
        a = random.choices(workers, weights=[w.weight for w in workers], k=1)[0]
        rest = [w for w in workers if w is not a]
        b = random.choices(rest, weights=[w.weight for w in rest], k=1)[0]
        return a if self._score(a) <= self._score(b) else b


def _p2c_from_config(config: "GatewayConfig") -> PowerOfTwoChoicesRouter:
    return PowerOfTwoChoicesRouter(
        config.etif_weight,
        config.effective_etif_scale,
        config.queue_weight,
        config.queue_scale,
        config.effective_metrics_max_age_s,
    )


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
        self._p2c = _p2c_from_config(config)

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
