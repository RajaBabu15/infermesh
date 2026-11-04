"""Routing test against real inference engines (llama.cpp), not a simulator.

N llama-server processes, a few KV slots each, sit behind the gateway. A workload of
shared-prefix groups (and a low-sharing control) is sent through the gateway with
cache-aware routing on and off. The engine's own counters report how many prompt
tokens it served from its prefix cache, which is the ground truth for hit rate.

  MODEL_GGUF=/path/to/model.gguf uv run python -m bench.e2e.routing --out /tmp/routing.json

"Off" is the same gateway with INFERMESH_CACHE_THRESHOLD=1.0, so routing falls back to
load-only power-of-two choices and the proxy overhead is identical in both arms.
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

WORDS = [w.strip() for w in open("/usr/share/dict/words") if w.strip().isalpha() and 3 <= len(w.strip()) <= 9]
GATEWAY_PORT = 8765
ENGINE_BASE_PORT = 8101


def words(seed: str, n: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n))


def wait_http(url: str, timeout_s: float = 120.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.5)
    raise RuntimeError(f"{url} did not come up")


def start_engines(model: str, n: int, ctx: int, slots: int) -> list[subprocess.Popen]:
    procs = []
    for i in range(n):
        log = open(f"/tmp/engine-{i}.log", "w")
        procs.append(subprocess.Popen(
            ["llama-server", "-m", model, "-c", str(ctx * slots), "-np", str(slots), "--port", str(ENGINE_BASE_PORT + i),
             "--metrics", "-ngl", "99", "--no-webui", "--cache-ram", "0"], stdout=log, stderr=log))
    for i in range(n):
        wait_http(f"http://127.0.0.1:{ENGINE_BASE_PORT + i}/health")
    return procs


def flush_caches(n: int, slots: int) -> None:
    """Overwrite every KV slot with an unrelated prompt so no earlier prefix survives
    (the host-memory prompt cache is off, see start_engines). Also warms the GPU."""
    for i in range(n):
        base = f"http://127.0.0.1:{ENGINE_BASE_PORT + i}"
        for k in range(slots):
            httpx.post(f"{base}/v1/chat/completions", timeout=300, json={
                "model": "m", "max_tokens": 1,
                "messages": [{"role": "user", "content": words(f"flush-{time.time()}-{i}-{k}", 80)}]})


def start_gateway(n: int, cache_threshold: float, extra_env: dict | None = None) -> subprocess.Popen:
    tmp = Path(tempfile.mkdtemp())
    (tmp / "workers.yaml").write_text("workers:\n" + "".join(
        f"  - {{id: e{i}, type: openai, url: 'http://127.0.0.1:{ENGINE_BASE_PORT + i}/v1', model: qwen, api_key_env: NO_KEY}}\n"
        for i in range(n)))
    env = {**os.environ, "INFERMESH_WORKERS_CONFIG_PATH": str(tmp / "workers.yaml"),
           "INFERMESH_CACHE_THRESHOLD": str(cache_threshold), "INFERMESH_HTTPX_TIMEOUT_S": "900",
           "INFERMESH_LOG_LEVEL": "WARNING", **(extra_env or {})}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "infermesh.main:create_app", "--factory",
                             "--port", str(GATEWAY_PORT), "--log-level", "warning"], env=env,
                            stdout=open("/tmp/gateway.log", "w"), stderr=subprocess.STDOUT)
    wait_http(f"http://127.0.0.1:{GATEWAY_PORT}/health")
    return proc


def workload(kind: str, groups: int, requests: int, prefix_words: int, seed: int) -> list[dict]:
    """kind 'shared': `groups` long prefixes, requests spread evenly and interleaved.
    kind 'unique': every request has its own prefix (a low-sharing control)."""
    rng = random.Random(seed)
    items = []
    for i in range(requests):
        g = i % groups if kind == "shared" else i
        system = words(f"{kind}-prefix-{g}", prefix_words)
        items.append({"model": "qwen", "max_tokens": 4, "temperature": 0,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": words(f"q-{seed}-{i}", 12)}]})
    rng.shuffle(items)
    return items


async def drive(items: list[dict], concurrency: int) -> tuple[list[dict], float]:
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = []

    async def one(client: httpx.AsyncClient, body: dict) -> None:
        async with sem:
            t0 = time.perf_counter()
            try:
                r = await client.post(f"http://127.0.0.1:{GATEWAY_PORT}/v1/chat/completions", json=body)
                dt = time.perf_counter() - t0
                if r.status_code == 200:
                    u = r.json()["usage"]
                    results.append({"ok": True, "latency_s": dt, "prompt_tokens": u["prompt_tokens"],
                                    "cached_tokens": u["prompt_tokens_details"]["cached_tokens"]})
                else:
                    results.append({"ok": False, "latency_s": dt, "status": r.status_code})
            except httpx.HTTPError as exc:
                results.append({"ok": False, "latency_s": time.perf_counter() - t0, "status": type(exc).__name__})

    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=900) as client:
        await asyncio.gather(*(one(client, b) for b in items))
    return results, time.perf_counter() - started


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    pos = (len(s) - 1) * q / 100
    lo = int(pos)
    return s[lo] + (s[min(lo + 1, len(s) - 1)] - s[lo]) * (pos - lo)


def summarize(results: list[dict], wall_s: float) -> dict:
    ok = [r for r in results if r["ok"]]
    lat = [r["latency_s"] for r in ok]
    prompt = sum(r["prompt_tokens"] for r in ok)
    cached = sum(r["cached_tokens"] for r in ok)
    return {"requests": len(results), "success_rate": len(ok) / len(results),
            "hit_ratio": cached / prompt if prompt else 0.0,
            "uncached_prompt_tokens": prompt - cached,
            "latency_p50_s": pct(lat, 50), "latency_p95_s": pct(lat, 95), "latency_p99_s": pct(lat, 99),
            "latency_mean_s": sum(lat) / len(lat), "throughput_rps": len(ok) / wall_s, "wall_s": wall_s}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("MODEL_GGUF"))
    ap.add_argument("--engines", type=int, default=4)
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--requests", type=int, default=48)
    ap.add_argument("--prefix-words", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--kinds", nargs="+", default=["shared", "unique"])
    ap.add_argument("--ctx", type=int, default=4096, help="context per slot")
    ap.add_argument("--slots", type=int, default=2, help="prefixes an engine can hold")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not args.model:
        sys.exit("set MODEL_GGUF or --model")

    engines = start_engines(args.model, args.engines, args.ctx, args.slots)
    runs: list[dict] = []
    try:
        flush_caches(args.engines, args.slots)
        for kind in args.kinds:
            for rep in range(args.reps):
                arms = [("cache_aware_on_completion", 0.3, {"INFERMESH_INSERT_ON_DISPATCH": "false"}),
                        ("cache_aware_on_dispatch", 0.3, {"INFERMESH_INSERT_ON_DISPATCH": "true"}),
                        ("load_only", 1.0, {})]
                arms = arms[rep % 3:] + arms[:rep % 3]       # rotate the order to cancel machine drift
                items = workload(kind, args.groups, args.requests, args.prefix_words, seed=rep)
                for arm, threshold, arm_env in arms:
                    flush_caches(args.engines, args.slots)
                    gw = start_gateway(args.engines, threshold, arm_env)
                    try:
                        results, wall = asyncio.run(drive(items, args.concurrency))
                    finally:
                        gw.terminate(); gw.wait(timeout=20)
                    row = {"kind": kind, "arm": arm, "rep": rep, **summarize(results, wall)}
                    runs.append(row)
                    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
                    Path(args.out).write_text(json.dumps(runs, indent=1))
    finally:
        for p in engines:
            p.terminate()


if __name__ == "__main__":
    main()
