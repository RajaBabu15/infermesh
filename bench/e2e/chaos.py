"""Failure injection against real processes: a gateway over three fake upstreams.

Open-loop load at a fixed rate; one fault is injected for a window and removed; the gateway
is judged on user-facing metrics (success rate, tail latency), how fast it recovers, and
how much extra upstream work it creates (attempts per request).

  uv run python -m bench.e2e.chaos --scenarios kill_worker total_outage

Scenarios: kill_worker (SIGKILL one upstream, restart it later), total_outage (every upstream
answers 503). A 429 storm and a slow worker were tried and dropped: with one repetition the two
arms disagreed, so the load was not reliably reaching the faulted worker. Arms: the retry budget off (a fixed two attempts) vs on (static 10 percent).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

from bench.e2e.routing import pct, wait_http

UP_BASE, GW_PORT = 9201, 8767


def start_upstream(i: int, latency_ms: int) -> subprocess.Popen:
    p = subprocess.Popen([sys.executable, "-m", "bench.e2e.fake_upstream", "--port", str(UP_BASE + i),
                          "--latency-ms", str(latency_ms)])
    wait_http(f"http://127.0.0.1:{UP_BASE + i}/health")
    return p


def start_gateway(n: int, env_extra: dict) -> subprocess.Popen:
    tmp = Path(tempfile.mkdtemp())
    (tmp / "workers.yaml").write_text("workers:\n" + "".join(
        f"  - {{id: u{i}, type: openai, url: 'http://127.0.0.1:{UP_BASE + i}/v1', model: m, api_key_env: NO_KEY}}\n"
        for i in range(n)))
    env = {**os.environ, "INFERMESH_WORKERS_CONFIG_PATH": str(tmp / "workers.yaml"), "INFERMESH_LOG_LEVEL": "WARNING", **env_extra}
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "infermesh.main:create_app", "--factory",
                          "--port", str(GW_PORT), "--log-level", "warning"], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wait_http(f"http://127.0.0.1:{GW_PORT}/health")
    return p


def set_fault(i: int, **q) -> None:
    httpx.post(f"http://127.0.0.1:{UP_BASE + i}/__fault", params=q, timeout=5)


def inject(scenario: str, ups: list, latency_ms: int, on: bool) -> None:
    if scenario == "kill_worker":
        if on:
            ups[0].kill(); ups[0].wait()
        else:
            ups[0] = start_upstream(0, latency_ms)
    elif scenario == "total_outage":
        for i in range(len(ups)):
            set_fault(i, **({"mode": "status", "code": 503} if on else {"mode": "ok"}))


async def run_load(rate: int, duration: float, fault_at: float, fault_for: float, scenario: str,
                   ups: list, latency_ms: int, n: int) -> dict:
    records: list[tuple[float, float, bool]] = []
    breaker: list[tuple[float, str]] = []
    t0 = time.perf_counter()

    async def one(client: httpx.AsyncClient, start: float, k: int) -> None:
        # Unique prompts: an identical prompt would pin every request to one worker via the prefix trie.
        body = {"model": "m", "max_tokens": 4, "messages": [{"role": "user", "content": f"request {k} {rng.random()}"}]}
        try:
            r = await client.post(f"http://127.0.0.1:{GW_PORT}/v1/chat/completions", json=body)
            ok = r.status_code == 200
        except httpx.HTTPError:
            ok = False
        records.append((start, time.perf_counter() - t0 - start, ok))

    async def watch() -> None:
        async with httpx.AsyncClient(timeout=2) as c:
            while time.perf_counter() - t0 < duration:
                try:
                    h = (await c.get(f"http://127.0.0.1:{GW_PORT}/health")).json()
                    state = next(w["circuit_state"] for w in h["worker_loads"] if w["id"] == "u0")
                    breaker.append((time.perf_counter() - t0, state))
                except (httpx.HTTPError, StopIteration, KeyError):
                    pass
                await asyncio.sleep(0.25)

    async def faults() -> None:
        await asyncio.sleep(fault_at)
        await asyncio.to_thread(inject, scenario, ups, latency_ms, True)
        await asyncio.sleep(fault_for)
        await asyncio.to_thread(inject, scenario, ups, latency_ms, False)

    sent = 0
    rng = random.Random(1)
    async with httpx.AsyncClient(timeout=30, limits=httpx.Limits(max_connections=1000)) as client:
        side = [asyncio.create_task(watch()), asyncio.create_task(faults())]
        tasks = []
        next_t = 0.0
        while next_t < duration:
            now = time.perf_counter() - t0
            if now < next_t:
                await asyncio.sleep(next_t - now)
            tasks.append(asyncio.create_task(one(client, next_t, sent)))
            sent += 1
            next_t += rng.expovariate(rate)                  # Poisson arrivals
        await asyncio.gather(*tasks, *side)
    received = 0
    for i in range(n):
        try:
            received += httpx.get(f"http://127.0.0.1:{UP_BASE + i}/__stats", timeout=3).json()["received"]
        except httpx.HTTPError:
            pass
    return {"records": records, "breaker": breaker, "sent": sent, "received": received}


def summarize(raw: dict, fault_at: float, fault_for: float) -> dict:
    recs = raw["records"]
    end = fault_at + fault_for
    window = [r for r in recs if fault_at <= r[0] < end]
    lat = lambda rows: [r[1] for r in rows if r[2]]
    succ = lambda rows: sum(r[2] for r in rows) / len(rows) if rows else float("nan")
    buckets: dict[int, list[bool]] = {}
    for start, _, ok in recs:
        buckets.setdefault(int(start), []).append(ok)
    recovered_at = None
    for sec in range(int(end), int(max(buckets) if buckets else end)):
        if all(buckets.get(s) and sum(buckets[s]) / len(buckets[s]) >= 0.99 for s in (sec, sec + 1, sec + 2)):
            recovered_at = sec
            break
    opened = next((t for t, s in raw["breaker"] if t >= fault_at and s == "OPEN"), None)
    return {"requests": len(recs), "success_rate": succ(recs), "window_success_rate": succ(window),
            "p50_ms": 1000 * pct(lat(recs), 50), "p99_ms": 1000 * pct(lat(recs), 99),
            "window_p99_ms": 1000 * pct(lat(window), 99) if lat(window) else float("nan"),
            "time_to_recover_s": (recovered_at - end) if recovered_at is not None else None,
            "breaker_open_after_s": (opened - fault_at) if opened is not None else None,
            "attempts_per_request": raw["received"] / max(1, raw["sent"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenarios", nargs="+", default=["kill_worker", "total_outage"])
    ap.add_argument("--rate", type=int, default=100)
    ap.add_argument("--duration", type=float, default=45)
    ap.add_argument("--fault-at", type=float, default=12)
    ap.add_argument("--fault-for", type=float, default=15)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--latency-ms", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n = 3
    arms = {"budget_off": {"INFERMESH_RETRY_BUDGET_MODE": "off"}, "budget_static": {"INFERMESH_RETRY_BUDGET_MODE": "static"}}
    rows = []
    for scenario in args.scenarios:
        for rep in range(args.reps):
            order = list(arms) if rep % 2 == 0 else list(reversed(list(arms)))
            for arm in order:
                ups = [start_upstream(i, args.latency_ms) for i in range(n)]
                gw = start_gateway(n, arms[arm])
                try:
                    raw = asyncio.run(run_load(args.rate, args.duration, args.fault_at, args.fault_for,
                                               scenario, ups, args.latency_ms, n))
                finally:
                    gw.terminate(); gw.wait(timeout=20)
                    for u in ups:
                        u.kill()
                row = {"scenario": scenario, "arm": arm, "rep": rep, **summarize(raw, args.fault_at, args.fault_for)}
                rows.append(row)
                print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
                Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
