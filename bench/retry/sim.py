"""E1: does the retry budget stop a metastable failure? (bench/retry)

Model (Bronson et al., "Metastable Failures in Distributed Systems", HotOS 2021, Sec 2.1):
a pool answers quickly while load is under its capacity, and its latency explodes above
it. Clients time out after `timeout_s` and retry. A timed-out attempt still occupies the
server (the server cannot know the client left), so retries add work that produces no
goodput. A short outage lets timeouts pile up; when service returns, the retries double
the load, latency stays above the timeout, and every attempt keeps failing.

  * N FIFO servers, each with service time `service_s` per attempt;
  * a request is sent to the server that owns its prefix (cache-aware affinity);
  * an attempt succeeds if its response arrives before its deadline;
  * during the outage attempts sent to the affected servers are lost (they only time out);
    by default every server is affected, with `outage_servers` only those are, so the
    others stay alive and fallback attempts can warm them;
  * after a timeout the gateway asks the retry budget (the real RetryBudget class) and,
    if admitted, sends one retry to a fallback server chosen by two random choices on queue length;
  * a server that already served a prefix holds it warm: an attempt on a warm server costs
    `service_s * (1 - warm_saving)` (the prefill share a prefix-cache hit saves).

Run: uv run python -m bench.retry.run
"""
from __future__ import annotations

import heapq
import itertools
import random
from dataclasses import dataclass, field

from infermesh.retry_budget import RetryBudget

_ARRIVE, _TIMEOUT, _DONE = 0, 1, 2


@dataclass
class Params:
    servers: int = 1
    capacity_rps: float = 300.0       # total capacity of the pool (Bronson: 300 QPS)
    load_rps: float = 280.0           # primary arrival rate (Bronson: 280 QPS)
    timeout_s: float = 1.0
    outage_start_s: float = 20.0
    outage_s: float = 10.0
    duration_s: float = 200.0
    max_attempts: int = 2
    prefixes: int = 20
    warm_saving: float = 0.0          # share of an attempt's work saved on a warm server
    outage_servers: tuple[int, ...] | None = None   # None = every server is hit by the outage


@dataclass
class Policy:
    name: str
    mode: str                         # "none" | "fixed" | "static" | "headroom"
    alpha: float = 0.0
    quota_margin: float = 0.95        # headroom mode: quota_rps = margin * capacity


@dataclass(eq=False)
class Request:
    rid: int
    prefix: int
    arrival_s: float
    attempts: int = 0
    done: bool = False


@dataclass(eq=False)
class Attempt:
    req: Request
    server: int
    deadline_s: float
    abandoned: bool = False


@dataclass
class Server:
    queue: list = field(default_factory=list)
    busy: bool = False
    warm: set = field(default_factory=set)


def run(params: Params, policy: Policy, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    now = 0.0
    clock = lambda: now
    if policy.mode == "none":
        budget = None
    elif policy.mode == "fixed":
        budget = RetryBudget(mode="off", clock=clock)
    else:
        budget = RetryBudget(
            mode="headroom" if policy.mode == "headroom" else "static",
            deposit=0.10, initial_credits=3.0, max_credits=10.0, alpha=policy.alpha,
            quota_rps=policy.quota_margin * params.capacity_rps, clock=clock)
    max_attempts = 1 if policy.mode == "none" else params.max_attempts
    per_server_s = params.servers / params.capacity_rps            # one server's cold service time
    servers = [Server() for _ in range(params.servers)]
    heap: list = []
    order = itertools.count()

    def push(t, kind, payload):
        heapq.heappush(heap, (t, next(order), kind, payload))

    t = 0.0
    rid = 0
    while t < params.duration_s:                                    # Poisson primary arrivals
        t += rng.expovariate(params.load_rps)
        push(t, _ARRIVE, Request(rid, rng.randrange(params.prefixes), t))
        rid += 1

    stats = dict(primaries=0, attempts=0, retries=0, denied=0, good=0)
    reqs: list[Request] = []
    good_times: list[float] = []

    def owner(prefix):
        return prefix % params.servers

    def in_outage(t_, server):
        if params.outage_servers is not None and server not in params.outage_servers:
            return False
        return params.outage_start_s <= t_ < params.outage_start_s + params.outage_s

    def service_time(att: Attempt) -> float:
        warm = att.req.prefix in servers[att.server].warm
        return per_server_s * (1.0 - params.warm_saving * (1.0 if warm else 0.0))

    def start_next(idx: int, t_: float) -> None:
        srv = servers[idx]
        if srv.busy or not srv.queue:
            return
        att = srv.queue.pop(0)
        srv.busy = True
        push(t_ + service_time(att), _DONE, (idx, att))

    def send(att: Attempt, t_: float) -> None:
        stats["attempts"] += 1
        push(att.deadline_s, _TIMEOUT, att)
        if in_outage(t_, att.server):
            return                                                  # lost: it will only time out
        servers[att.server].queue.append(att)
        start_next(att.server, t_)

    while heap:
        now, _, kind, payload = heapq.heappop(heap)
        if now > params.duration_s + params.timeout_s * 3:
            break
        if kind == _ARRIVE:
            req: Request = payload
            stats["primaries"] += 1
            reqs.append(req)
            if budget is not None:
                budget.on_request()
            req.attempts = 1
            send(Attempt(req, owner(req.prefix), now + params.timeout_s), now)
        elif kind == _DONE:
            idx, att = payload
            srv = servers[idx]
            srv.busy = False
            srv.warm.add(att.req.prefix)
            if not att.abandoned and not att.req.done:
                att.req.done = True
                stats["good"] += 1
                good_times.append(now)
            start_next(idx, now)
        else:                                                       # TIMEOUT
            att: Attempt = payload
            if att.req.done or att.abandoned:
                continue
            att.abandoned = True
            req = att.req
            if req.attempts >= max_attempts:
                continue
            others = [i for i in range(params.servers) if i != att.server] or [att.server]
            a, b = rng.choice(others), rng.choice(others)
            fallback = a if len(servers[a].queue) <= len(servers[b].queue) else b
            hit = 1.0 if req.prefix in servers[fallback].warm else 0.0
            if budget is not None and not budget.try_acquire(hit):
                stats["denied"] += 1
                continue
            stats["retries"] += 1
            req.attempts += 1
            send(Attempt(req, fallback, now + params.timeout_s), now)

    end = params.outage_start_s + params.outage_s
    window = [g for g in good_times if end + 30.0 <= g < params.duration_s]
    tail_rate = len(window) / max(1e-9, params.duration_s - end - 30.0)
    recovered = tail_rate >= 0.9 * params.load_rps
    after = [g for g in good_times if g >= end]
    in_window = [r for r in reqs if params.outage_start_s <= r.arrival_s < end]
    return {
        "window_success_ratio": sum(r.done for r in in_window) / max(1, len(in_window)),
        "window_attempts_per_request": sum(r.attempts for r in in_window) / max(1, len(in_window)),
        "goodput_after_outage_rps": len(after) / max(1e-9, params.duration_s - end),
        "steady_goodput_rps": tail_rate,
        "recovered": float(recovered),
        "success_ratio": stats["good"] / max(1, stats["primaries"]),
        "attempts_per_request": stats["attempts"] / max(1, stats["primaries"]),
        "retries_denied": float(stats["denied"]),
    }
