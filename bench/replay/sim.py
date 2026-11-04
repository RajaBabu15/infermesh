"""Deterministic discrete-event replay of routing policies on a prefix-reuse trace.

Cluster model (an assumption, not a measurement of any real GPU):
  * each worker has one FIFO prefill server and unlimited concurrent decodes;
  * a worker's engine keeps an LRU cache of 512-token KV blocks (capacity
    `kv_blocks`); a request's cached tokens are the consecutive leading shared
    blocks present at prefill start; blocks are committed when prefill ends;
  * prefill time follows the form in DistServe Appendix A (arXiv 2401.09670):
    a term linear in the uncached tokens, a term quadratic in them, and a constant.
    DistServe fits those constants per model and GPU by profiling and publishes
    none, so ServiceModel's constants are ASSUMPTIONS. Claim only the relative
    ordering of policies, never absolute latencies.

The policies that use the gateway's own code are the real classes:
KVAwareRouter and PowerOfTwoChoicesRouter from infermesh.routing and RadixTrie from
infermesh.trie, fed character strings synthesized from the trace (see bench/replay/trace.py).
"""
from __future__ import annotations

import bisect
import hashlib
import heapq
import itertools
import math
import random
from collections import OrderedDict, deque
from dataclasses import dataclass, field

from bench.replay.metrics import RequestResult
from bench.replay.trace import BLOCK_TOKENS, CHARS_PER_TOKEN, TraceRequest, synthesize_prompt
from infermesh.routing import KVAwareRouter, PowerOfTwoChoicesRouter
from infermesh.trie import RadixTrie
from infermesh.settings import GatewayConfig, WorkerConfig

MODEL = "m"
DEFAULT_OUTPUT_TOKENS = 256        # what the gateway reserves for output (_estimate_tokens default)


@dataclass(frozen=True)
class ServiceModel:
    c1: float = 4e-5               # seconds per uncached prompt token   (assumed)
    c2: float = 3e-9               # seconds per uncached token squared  (assumed)
    c3: float = 0.02               # fixed seconds per prefill           (assumed)
    itl: float = 0.02              # seconds per output token after the first (assumed)

    def prefill_s(self, uncached_tokens: int) -> float:
        u = max(0, uncached_tokens)
        return self.c1 * u + self.c2 * u * u + self.c3


class Engine:
    """LRU cache of 512-token KV blocks."""

    def __init__(self, kv_blocks: int) -> None:
        self.kv_blocks = kv_blocks
        self._lru: OrderedDict[int, None] = OrderedDict()

    def cached_blocks(self, block_ids: tuple[int, ...]) -> int:
        n = 0
        for block in block_ids:
            if block not in self._lru:
                break
            n += 1
        return n

    def commit(self, block_ids: tuple[int, ...]) -> None:
        for block in block_ids:
            self._lru.pop(block, None)
            self._lru[block] = None
        while len(self._lru) > self.kv_blocks:
            self._lru.popitem(last=False)


@dataclass(eq=False)
class Job:
    req: TraceRequest
    arrival_s: float
    worker: int = -1
    cached_tokens: int = 0
    ttft_s: float = 0.0
    est_tokens: int = 0
    _prompt: str | None = None

    @property
    def prompt(self) -> str:
        if self._prompt is None:
            self._prompt = synthesize_prompt(self.req)
        return self._prompt


@dataclass(eq=False)
class Worker:
    idx: int
    cfg: WorkerConfig
    engine: Engine
    queue: deque = field(default_factory=deque)
    current: Job | None = None
    busy_until: float = 0.0


class Cluster:
    """What a policy may look at: live gateway-side load, and a possibly stale
    snapshot of each engine's prefill backlog (the quantity an engine metric
    such as vllm:num_requests_waiting stands in for)."""

    def __init__(self, workers: list[Worker], service: ServiceModel, staleness_s: float) -> None:
        self.workers = workers
        self.service = service
        self.staleness_s = staleness_s
        self.now = 0.0
        self.arrivals = 0
        self._snap: list[float] = [0.0] * len(workers)
        self._snap_at = -math.inf

    def backlog_s(self, w: Worker) -> float:
        """Seconds until worker `w` could start a request arriving now (live)."""
        busy = max(0.0, w.busy_until - self.now) if w.current is not None else 0.0
        queued = sum(
            self.service.prefill_s(j.req.input_len - w.engine.cached_blocks(j.req.shared_block_ids) * BLOCK_TOKENS)
            for j in w.queue
        )
        return busy + queued

    def stale_backlog_s(self, w: Worker) -> float:
        if self.staleness_s <= 0:
            return self.backlog_s(w)
        if self.now - self._snap_at >= self.staleness_s:
            self._snap = [self.backlog_s(x) for x in self.workers]
            self._snap_at = self.now
        return self._snap[w.idx]


class Policy:
    stochastic = False

    def attach(self, cluster: Cluster, rng: random.Random) -> None:
        self.cluster = cluster
        self.rng = rng

    def choose(self, job: Job) -> int:
        raise NotImplementedError

    def on_finish(self, job: Job) -> None:
        pass


class RoundRobin(Policy):
    def __init__(self) -> None:
        self._next = itertools.count()

    def choose(self, job: Job) -> int:
        return next(self._next) % len(self.cluster.workers)


class LeastLoaded(Policy):
    def choose(self, job: Job) -> int:
        ws = self.cluster.workers
        return min(range(len(ws)), key=lambda i: (ws[i].cfg.requests_in_flight, i))


class P2C(Policy):
    """The gateway's PowerOfTwoChoicesRouter on tokens in flight (ETIF)."""

    stochastic = True

    def __init__(self, etif_scale: int = 200_000) -> None:
        self._router = PowerOfTwoChoicesRouter(etif_weight=0.6, etif_scale=etif_scale)

    def choose(self, job: Job) -> int:
        picked = self._router.pick([w.cfg for w in self.cluster.workers])
        return next(w.idx for w in self.cluster.workers if w.cfg is picked)


class ConsistentHash(Policy):
    """Consistent hashing on the first shared block (or the request, if none)."""

    def __init__(self, vnodes: int = 160) -> None:
        self._vnodes = vnodes
        self._ring: list[tuple[int, int]] = []

    @staticmethod
    def _h(text: str) -> int:
        return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")

    def attach(self, cluster: Cluster, rng: random.Random) -> None:
        super().attach(cluster, rng)
        self._ring = sorted((self._h(f"w{w.idx}-{v}"), w.idx) for w in cluster.workers for v in range(self._vnodes))
        self._keys = [k for k, _ in self._ring]

    def choose(self, job: Job) -> int:
        shared = job.req.shared_block_ids
        key = self._h(f"b{shared[0]}" if shared else f"r{job.req.index}")
        pos = bisect.bisect(self._keys, key) % len(self._ring)
        return self._ring[pos][1]


class _RegistryStub:
    def __init__(self, configs: list[WorkerConfig]) -> None:
        self._configs = configs

    def get_workers(self, model: str) -> list[WorkerConfig]:
        return self._configs


def _trie() -> RadixTrie:
    return RadixTrie(ttl_s=10 ** 9, max_chars_per_worker=2 ** 26)


class CacheAware(Policy):
    """The gateway's real KVAwareRouter over a real RadixTrie."""

    stochastic = True

    def __init__(self, etif_scale: int = 200_000, **config) -> None:
        self._config = GatewayConfig(_env_file=None, etif_scale=etif_scale, **config)

    def attach(self, cluster: Cluster, rng: random.Random) -> None:
        super().attach(cluster, rng)
        self.trie = _trie()
        self._router = KVAwareRouter(_RegistryStub([w.cfg for w in cluster.workers]), self.trie, self._config)

    def choose(self, job: Job) -> int:
        picked, _ = self._router.select(MODEL, [], prefix=job.prompt)
        return next(w.idx for w in self.cluster.workers if w.cfg is picked)

    def on_finish(self, job: Job) -> None:
        self.trie.insert(MODEL, job.prompt, self.cluster.workers[job.worker].cfg.id)


class ExpectedTTFT(Policy):
    """Score every worker by expected TTFT = (possibly stale) prefill backlog +
    prefill time of the prompt's uncached part, using the gateway's live trie to
    estimate each worker's cached part.

      argmin   lowest expected TTFT (herds when the backlog is stale)
      p2c      two random workers, lower expected TTFT
      softmax  P(i) ~ exp(-cost_i / tau), tau = c * lambda_w * T * s_bar:
               the backlog can drift by lambda_w * T arrivals of s_bar seconds
               each while the snapshot is T seconds old (E2, a hypothesis).
    """

    stochastic = True

    def __init__(self, mode: str, c: float = 1.0) -> None:
        self.mode = mode
        self.c = c
        self._s_sum = 0.0
        self._s_n = 0

    def attach(self, cluster: Cluster, rng: random.Random) -> None:
        super().attach(cluster, rng)
        self.trie = _trie()

    def _costs(self, job: Job) -> list[float]:
        costs = []
        for w in self.cluster.workers:
            matched_chars = self.trie.longest_prefix_match(MODEL, job.prompt, {w.cfg.id})[1]
            uncached = max(0, job.req.input_len - matched_chars // CHARS_PER_TOKEN)
            costs.append(self.cluster.stale_backlog_s(w) + self.cluster.service.prefill_s(uncached))
        return costs

    def choose(self, job: Job) -> int:
        costs = self._costs(job)
        s = self.cluster.service.prefill_s(job.req.input_len)
        self._s_sum += s
        self._s_n += 1
        n = len(costs)
        if self.mode == "p2c":
            a, b = self.rng.sample(range(n), 2) if n > 1 else (0, 0)
            return a if costs[a] <= costs[b] else b
        if self.mode == "softmax":
            now = max(self.cluster.now, 1.0)
            lam_w = self.cluster.arrivals / now / n
            tau = self.c * lam_w * self.cluster.staleness_s * (self._s_sum / self._s_n)
            if tau > 1e-12:
                low = min(costs)
                weights = [math.exp(-(c - low) / tau) for c in costs]
                return self.rng.choices(range(n), weights=weights, k=1)[0]
        return min(range(n), key=lambda i: (costs[i], i))

    def on_finish(self, job: Job) -> None:
        self.trie.insert(MODEL, job.prompt, self.cluster.workers[job.worker].cfg.id)


POLICIES = {
    "round_robin": lambda **_: RoundRobin(),
    "least_loaded": lambda **_: LeastLoaded(),
    "p2c": lambda **_: P2C(),
    "consistent_hash": lambda **_: ConsistentHash(),
    "cache_aware": lambda **_: CacheAware(),
    "ttft_argmin": lambda **_: ExpectedTTFT("argmin"),
    "ttft_p2c": lambda **_: ExpectedTTFT("p2c"),
    "ttft_softmax": lambda **kw: ExpectedTTFT("softmax", c=kw.get("softmax_c", 1.0)),
}

_ARRIVE, _PREFILL_DONE, _FINISH = 0, 1, 2


def simulate(
    requests: list[TraceRequest],
    policy: str,
    seed: int = 0,
    workers: int = 4,
    kv_blocks: int = 4000,
    speedup: float = 1.0,
    staleness_s: float = 0.0,
    service: ServiceModel = ServiceModel(),
    **policy_kwargs,
) -> list[RequestResult]:
    random.seed(seed)                          # the gateway's routers draw from the global RNG
    rng = random.Random(seed)
    pool = [Worker(i, WorkerConfig(id=f"w{i}", type="openai", url="http://sim", model=MODEL),
                   Engine(kv_blocks)) for i in range(workers)]
    cluster = Cluster(pool, service, staleness_s)
    pol = POLICIES[policy](**policy_kwargs)
    pol.attach(cluster, rng)

    heap: list = []
    order = itertools.count()
    for req in requests:
        heapq.heappush(heap, (req.ts_s / speedup, next(order), _ARRIVE, req))
    results: list[RequestResult] = []

    def start_prefill(w: Worker, now: float) -> None:
        job = w.queue.popleft()
        job.cached_tokens = min(job.req.input_len,
                                w.engine.cached_blocks(job.req.shared_block_ids) * BLOCK_TOKENS)
        duration = service.prefill_s(job.req.input_len - job.cached_tokens)
        w.current = job
        w.busy_until = now + duration
        heapq.heappush(heap, (now + duration, next(order), _PREFILL_DONE, w.idx))

    while heap:
        now, _, kind, payload = heapq.heappop(heap)
        cluster.now = now
        if kind == _ARRIVE:
            req: TraceRequest = payload
            cluster.arrivals += 1
            job = Job(req, arrival_s=now, est_tokens=req.input_len + DEFAULT_OUTPUT_TOKENS)
            job.worker = pol.choose(job)
            w = pool[job.worker]
            w.cfg.reserve(job.est_tokens)
            w.queue.append(job)
            if w.current is None:
                start_prefill(w, now)
        elif kind == _PREFILL_DONE:
            w = pool[payload]
            job = w.current
            w.engine.commit(job.req.shared_block_ids)
            job.ttft_s = now - job.arrival_s
            w.current = None
            heapq.heappush(heap, (now + max(0, job.req.output_len - 1) * service.itl,
                                  next(order), _FINISH, job))
            if w.queue:
                start_prefill(w, now)
        else:
            job = payload
            pool[job.worker].cfg.release(job.est_tokens)
            pol.on_finish(job)
            results.append(RequestResult(
                index=job.req.index, arrival_s=job.arrival_s, ttft_s=job.ttft_s,
                e2e_s=now - job.arrival_s, input_len=job.req.input_len,
                output_len=job.req.output_len, cached_tokens=job.cached_tokens, worker=job.worker))
    results.sort(key=lambda r: r.index)
    return results
