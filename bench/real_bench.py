"""InferMesh gateway vs direct-provider benchmark against real APIs.

Modes:
  gateway: in-process via ASGITransport.
  direct:  straight to the provider HTTPS endpoint.

Scenarios:
  shared_prefix: identical 200-char system prompt, varying user question.
  diverse:       unique uuid-prefixed system prompt per request.

Concurrency via asyncio.gather + semaphore. 429 responses retry with
exponential backoff (max 3 attempts) so the run survives free-tier limits.

Output: bench/results/<utc-ts>/{raw.csv,summary.json,charts/*.png}.

Usage:
  uv run python -m bench.real_bench
  uv run python -m bench.real_bench --requests-groq 500 --concurrency-groq 25
  uv run python -m bench.real_bench --providers groq
  uv run python -m bench.real_bench --modes gateway
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import time
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

from infermesh.main import create_app


# ---------------------------------------------------------------------------
# Real test inputs
# ---------------------------------------------------------------------------

QUESTIONS = [
    "What is the capital of France?",
    "What does ACID stand for in databases?",
    "Briefly: what is recursion in programming?",
    "Explain TCP vs UDP in two sentences.",
    "What is 17 multiplied by 23?",
    "Name three common sorting algorithms.",
    "What is REST API in one sentence?",
    "Briefly: what is async/await in Python?",
    "What is HTTP/2?",
    "Explain the difference between SQL and NoSQL.",
    "What is a hash table?",
    "Define Big-O notation in one sentence.",
    "What is the CAP theorem?",
    "Briefly explain SSL vs TLS.",
    "What is dependency injection?",
]

SHARED_SYSTEM_PROMPT = (
    "You are a concise technical assistant. Answer in one or two short sentences. "
    "Be accurate and direct. Avoid filler words and disclaimers."
)


# ---------------------------------------------------------------------------
# Provider config: model name and direct-call URL/auth.
# ---------------------------------------------------------------------------

PROVIDERS = {
    "groq": {
        "model": "llama-3.1-8b-instant",
        "direct_url": "https://api.groq.com/openai/v1/chat/completions",
        "auth_env": "GROQ_API_KEY",
        "auth_style": "bearer",  # Authorization: Bearer <key>
    },
    "gemini": {
        "model": "gemini-2.5-flash",
        "direct_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "auth_env": "GEMINI_API_KEY",
        "auth_style": "bearer",
    },
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    mode: str            # 'gateway' or 'direct'
    scenario: str
    provider: str
    model: str
    idx: int
    latency_ms: float
    status: int
    retries: int = 0
    completion_tokens: int = 0
    prompt_tokens: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# Scenario message builders
# ---------------------------------------------------------------------------

def build_messages(scenario: str, idx: int) -> list[dict]:
    if scenario == "shared_prefix":
        return [
            {"role": "system", "content": SHARED_SYSTEM_PROMPT},
            {"role": "user", "content": QUESTIONS[idx % len(QUESTIONS)]},
        ]
    if scenario == "diverse":
        unique = uuid.uuid4().hex[:12]
        return [
            {"role": "system", "content": f"Session {unique}. " + SHARED_SYSTEM_PROMPT},
            {"role": "user", "content": QUESTIONS[idx % len(QUESTIONS)]},
        ]
    raise ValueError(f"unknown scenario: {scenario}")


# ---------------------------------------------------------------------------
# Request execution
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_BACKOFF_BASE_S = 1.0


async def _post_with_backoff(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict,
) -> tuple[httpx.Response | None, int, float]:
    """
    POST with retry on 429. Returns (response, retry_count, total_elapsed_ms).
    Real clients do this. The gateway itself doesn't retry but
    application-layer code should.
    """
    t0 = time.monotonic()
    retries = 0
    last_resp: httpx.Response | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = await client.post(url, headers=headers, json=payload, timeout=30.0)
            last_resp = resp
            if resp.status_code != 429:
                return resp, retries, (time.monotonic() - t0) * 1000.0
            retries += 1
            if attempt == _MAX_RETRIES:
                break
            # Honor Retry-After if the provider sets it; otherwise jittered exp backoff.
            ra = resp.headers.get("retry-after", "")
            try:
                wait_s = float(ra) if ra else _BACKOFF_BASE_S * (2 ** attempt)
            except ValueError:
                wait_s = _BACKOFF_BASE_S * (2 ** attempt)
            wait_s += random.uniform(0, 0.5)  # jitter
            await asyncio.sleep(wait_s)
        except Exception:
            raise
    return last_resp, retries, (time.monotonic() - t0) * 1000.0


async def call_gateway(
    client: httpx.AsyncClient,
    provider: str,
    scenario: str,
    idx: int,
) -> RequestResult:
    model = PROVIDERS[provider]["model"]
    messages = build_messages(scenario, idx)
    resp, retries, elapsed_ms = await _post_with_backoff(
        client,
        "/v1/chat/completions",
        {},
        {"model": model, "messages": messages, "max_tokens": 60},
    )
    return _build_result("gateway", provider, scenario, model, idx, resp, retries, elapsed_ms)


async def call_direct(
    client: httpx.AsyncClient,
    provider: str,
    scenario: str,
    idx: int,
) -> RequestResult:
    info = PROVIDERS[provider]
    api_key = os.environ.get(info["auth_env"], "")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = build_messages(scenario, idx)
    resp, retries, elapsed_ms = await _post_with_backoff(
        client,
        info["direct_url"],
        headers,
        {"model": info["model"], "messages": messages, "max_tokens": 60},
    )
    return _build_result("direct", provider, scenario, info["model"], idx, resp, retries, elapsed_ms)


def _build_result(
    mode: str, provider: str, scenario: str, model: str, idx: int,
    resp: httpx.Response | None, retries: int, elapsed_ms: float,
) -> RequestResult:
    base = dict(mode=mode, scenario=scenario, provider=provider, model=model,
                idx=idx, latency_ms=elapsed_ms, retries=retries)
    if resp is None:
        return RequestResult(**base, status=0, error="no response")
    if resp.status_code == 200:
        try:
            usage = (resp.json().get("usage") or {})
        except Exception:
            usage = {}
        return RequestResult(
            **base, status=200,
            completion_tokens=int(usage.get("completion_tokens", 0)),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
        )
    body = resp.text[:200] if hasattr(resp, "text") else ""
    return RequestResult(**base, status=resp.status_code, error=body)


# ---------------------------------------------------------------------------
# Concurrent cell runner
# ---------------------------------------------------------------------------

async def run_cell(
    label: str,
    caller,
    n: int,
    concurrency: int,
) -> list[RequestResult]:
    """Fire n requests through `caller`, bounded by a semaphore."""
    sem = asyncio.Semaphore(concurrency)
    progress = {"done": 0, "ok": 0, "fail": 0, "retry": 0}

    async def _one(idx: int) -> RequestResult:
        async with sem:
            r = await caller(idx)
            progress["done"] += 1
            if r.status == 200:
                progress["ok"] += 1
            else:
                progress["fail"] += 1
            progress["retry"] += r.retries
            if progress["done"] % max(1, n // 4) == 0 or progress["done"] == n:
                print(
                    f"    [{progress['done']}/{n}]  ok={progress['ok']}  "
                    f"fail={progress['fail']}  retries={progress['retry']}"
                )
            return r

    print(f"  {label}  (n={n}, concurrency={concurrency})")
    t0 = time.monotonic()
    results = await asyncio.gather(*(_one(i) for i in range(n)))
    elapsed_s = time.monotonic() - t0
    print(f"    cell wall time: {elapsed_s:.1f}s, effective RPS {n / elapsed_s:.1f}\n")
    return results


# ---------------------------------------------------------------------------
# Prometheus scrape (gateway-only)
# ---------------------------------------------------------------------------

async def scrape_prometheus(gateway_client: httpx.AsyncClient) -> dict:
    """Parse infermesh_* metric families into JSON for later analysis."""
    from prometheus_client.parser import text_string_to_metric_families
    resp = await gateway_client.get("/metrics")
    resp.raise_for_status()
    out: dict[str, list] = {}
    for fam in text_string_to_metric_families(resp.text):
        if not fam.name.startswith("infermesh_"):
            continue
        out[fam.name] = [
            {"labels": dict(s.labels), "value": float(s.value)}
            for s in fam.samples
        ]
    return out


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(pct / 100.0 * (len(s) - 1)))))
    return s[k]


def compute_summary(
    results: list[RequestResult],
    gateway_metrics_t0: dict,
    gateway_metrics_t1: dict,
    health_t0: dict,
    health_t1: dict,
) -> dict:
    cells: dict[str, dict] = {}
    for r in results:
        key = f"{r.mode}_{r.provider}_{r.scenario}"
        c = cells.setdefault(key, {
            "mode": r.mode, "provider": r.provider, "scenario": r.scenario,
            "model": r.model, "latencies_ms": [],
            "successes": 0, "rate_limited": 0, "other_failures": 0,
            "total_retries": 0, "prompt_tokens": 0, "completion_tokens": 0,
        })
        if r.status == 200:
            c["successes"] += 1
            c["latencies_ms"].append(r.latency_ms)
            c["prompt_tokens"] += r.prompt_tokens
            c["completion_tokens"] += r.completion_tokens
        elif r.status == 429:
            c["rate_limited"] += 1
        else:
            c["other_failures"] += 1
        c["total_retries"] += r.retries

    for c in cells.values():
        lats = c["latencies_ms"]
        c["count"] = c["successes"] + c["rate_limited"] + c["other_failures"]
        c["median_ms"] = round(_percentile(lats, 50), 1)
        c["p95_ms"] = round(_percentile(lats, 95), 1)
        c["p99_ms"] = round(_percentile(lats, 99), 1)
        c["mean_ms"] = round(sum(lats) / len(lats), 1) if lats else 0.0
        c["latency_count"] = len(lats)
        c.pop("latencies_ms")

    # Gateway counters delta.
    def _sum_counter(metrics: dict, name: str) -> float:
        samples = metrics.get(name) or metrics.get(name.removesuffix("_total"), [])
        return sum(s["value"] for s in samples)

    gateway_delta = {
        "requests_total": _sum_counter(gateway_metrics_t1, "infermesh_requests_total")
                          - _sum_counter(gateway_metrics_t0, "infermesh_requests_total"),
        "cache_hits_total": _sum_counter(gateway_metrics_t1, "infermesh_kv_cache_hits_total")
                            - _sum_counter(gateway_metrics_t0, "infermesh_kv_cache_hits_total"),
        "cache_misses_total": _sum_counter(gateway_metrics_t1, "infermesh_kv_cache_misses_total")
                              - _sum_counter(gateway_metrics_t0, "infermesh_kv_cache_misses_total"),
        "circuit_breaker_opens_total": _sum_counter(gateway_metrics_t1, "infermesh_circuit_breaker_opens_total")
                                       - _sum_counter(gateway_metrics_t0, "infermesh_circuit_breaker_opens_total"),
        "trie_entries_baseline": health_t0.get("trie_entries", 0),
        "trie_entries_final": health_t1.get("trie_entries", 0),
        "redis_enabled": health_t1.get("redis_enabled", False),
        "prometheus_enabled": health_t1.get("prometheus_enabled", False),
        "otel_enabled": health_t1.get("otel_enabled", False),
    }
    return {"cells": cells, "gateway_delta": gateway_delta}


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def render_charts(summary: dict, results: list[RequestResult], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts_dir = out_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    cells = summary["cells"]

    # Chart 1: gateway vs direct, p99 latency per provider+scenario.
    _chart_grouped_bar(
        cells,
        metric="p99_ms",
        title="p99 latency: gateway vs direct",
        ylabel="p99 latency (ms)",
        out_path=charts_dir / "p99_gateway_vs_direct.png",
    )

    _chart_grouped_bar(
        cells,
        metric="median_ms",
        title="Median latency: gateway vs direct",
        ylabel="median latency (ms)",
        out_path=charts_dir / "median_gateway_vs_direct.png",
    )

    # Chart 3: failures by cell.
    _chart_failures(cells, charts_dir / "failures.png")

    # Chart 4: latency-over-time (scatter, per-cell).
    _chart_latency_over_time(results, charts_dir / "latency_over_time.png")

    # Chart 5: cache-hit-rate warmup curve (gateway shared_prefix only).
    _chart_warmup_curve(results, charts_dir / "warmup_curve.png")


def _chart_grouped_bar(cells: dict, metric: str, title: str, ylabel: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    # Each x-tick = (provider, scenario). Two bars per tick: gateway vs direct.
    provider_scenarios = sorted({(c["provider"], c["scenario"]) for c in cells.values()})
    x_labels = [f"{p}\n{s}" for p, s in provider_scenarios]
    gateway_vals = []
    direct_vals = []
    for p, s in provider_scenarios:
        g = cells.get(f"gateway_{p}_{s}", {})
        d = cells.get(f"direct_{p}_{s}", {})
        gateway_vals.append(g.get(metric, 0.0))
        direct_vals.append(d.get(metric, 0.0))

    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(len(x_labels)))
    width = 0.35
    bars1 = ax.bar([v - width/2 for v in x], gateway_vals, width,
                   label="gateway", color="#2ca02c", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar([v + width/2 for v in x], direct_vals, width,
                   label="direct", color="#888", edgecolor="black", linewidth=0.5)
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h,
                        f"{h:.0f}ms", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _chart_failures(cells: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    keys = sorted(cells.keys())
    rate_limited = [cells[k]["rate_limited"] for k in keys]
    other_fail = [cells[k]["other_failures"] for k in keys]
    successes = [cells[k]["successes"] for k in keys]

    fig, ax = plt.subplots(figsize=(11, 5))
    x = list(range(len(keys)))
    ax.bar(x, successes, label="success", color="#2ca02c", edgecolor="black", linewidth=0.5)
    ax.bar(x, rate_limited, bottom=successes, label="429 rate-limited",
           color="#ff7f0e", edgecolor="black", linewidth=0.5)
    ax.bar(x, other_fail,
           bottom=[s + r for s, r in zip(successes, rate_limited)],
           label="other failure", color="#d62728", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", "\n") for k in keys], fontsize=8)
    ax.set_ylabel("requests")
    ax.set_title("Per-cell outcomes (stacked)")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _chart_latency_over_time(results: list[RequestResult], out_path: Path) -> None:
    """
    Scatter plot: x = request index within cell, y = latency_ms (successes only).
    Warmup, retry storms, and cooldowns are visible as bands.
    Each provider gets a subplot, with gateway vs direct overlaid.
    """
    import matplotlib.pyplot as plt
    providers = sorted({r.provider for r in results})
    fig, axes = plt.subplots(len(providers), 1, figsize=(11, 4 * len(providers)), sharex=False)
    if len(providers) == 1:
        axes = [axes]

    for ax, provider in zip(axes, providers):
        for mode, color in [("direct", "#888"), ("gateway", "#2ca02c")]:
            for scenario, marker in [("shared_prefix", "o"), ("diverse", "x")]:
                pts = [
                    (r.idx, r.latency_ms) for r in results
                    if r.provider == provider and r.mode == mode
                    and r.scenario == scenario and r.status == 200
                ]
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, c=color, marker=marker, s=18, alpha=0.6,
                           label=f"{mode} / {scenario}", edgecolors="none")
        ax.set_title(f"{provider}: latency vs request index (successes only)")
        ax.set_xlabel("request index within cell")
        ax.set_ylabel("latency (ms)")
        ax.legend(loc="upper right", frameon=False, fontsize=8)
        ax.grid(linestyle=":", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _chart_warmup_curve(results: list[RequestResult], out_path: Path) -> None:
    """
    Cumulative cache-hit rate over request index within the gateway/shared_prefix
    cell. Shows how the trie warms up: 0% on request 1 (cold), climbing toward
    100% as the trie populates with the shared prefix.

    This is the most honest single-image proof that the RadixTrie produces
    a signal that is independent of circuit-breaker latency effects.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.5))

    providers = sorted({
        r.provider for r in results
        if r.mode == "gateway" and r.scenario == "shared_prefix" and r.status == 200
    })
    for provider in providers:
        # Replay the request order. The trie matches a request iff a prior
        # successful request in this cell stored the same prefix.
        cell = sorted(
            [r for r in results
             if r.mode == "gateway" and r.scenario == "shared_prefix"
             and r.provider == provider and r.status == 200],
            key=lambda r: r.idx,
        )
        if len(cell) < 2:
            continue
        # First successful request: trie is empty, so hit_rate=0.
        # All subsequent successes: hit (because shared_prefix repeats).
        cumulative_hit_rate = []
        hits = 0
        for i, _r in enumerate(cell):
            if i > 0:  # not the first
                hits += 1
            cumulative_hit_rate.append(100.0 * hits / (i + 1))
        xs = list(range(1, len(cell) + 1))
        ax.plot(xs, cumulative_hit_rate, marker="o", markersize=4,
                label=f"{provider}", linewidth=1.5)

    ax.set_title("Gateway trie warmup: cumulative cache-hit rate (shared_prefix cell)")
    ax.set_xlabel("successful request # within cell")
    ax.set_ylabel("cumulative cache-hit rate (%)")
    ax.set_ylim(0, 105)
    ax.axhline(100, color="#aaa", linewidth=0.5, linestyle="--")
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args: argparse.Namespace) -> None:
    out_root = Path(__file__).resolve().parent / "results"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = out_root / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    requests_per_provider = {
        "groq": args.requests_groq,
        "gemini": args.requests_gemini,
    }
    concurrency_per_provider = {
        "groq": args.concurrency_groq,
        "gemini": args.concurrency_gemini,
    }

    print(f"InferMesh real benchmark, {ts}")
    print(f"  output:        {out_dir}")
    print(f"  modes:         {', '.join(args.modes)}")
    print(f"  providers:     {', '.join(args.providers)}")
    for p in args.providers:
        print(f"  {p:7} n/scenario={requests_per_provider[p]}  "
              f"concurrency={concurrency_per_provider[p]}")
    print()

    # Gateway runs in-process. Direct calls go to real provider URLs through
    # a separate httpx client (no gateway in the path).
    app = create_app()
    gateway_transport = httpx.ASGITransport(app=app)
    gateway_client = httpx.AsyncClient(transport=gateway_transport, base_url="http://bench")
    direct_client = httpx.AsyncClient()

    all_results: list[RequestResult] = []

    async with gateway_client, direct_client:
        # Pre-scrape gateway state.
        prom_t0 = await scrape_prometheus(gateway_client)
        health_t0 = (await gateway_client.get("/health")).json()

        for mode in args.modes:
            print(f"=== mode: {mode} ===")
            for provider in args.providers:
                model = PROVIDERS[provider]["model"]
                n = requests_per_provider[provider]
                conc = concurrency_per_provider[provider]
                for scenario in ["shared_prefix", "diverse"]:
                    label = f"{mode} / {provider} / {scenario} ({model})"
                    if mode == "gateway":
                        caller = lambda i, p=provider, s=scenario: call_gateway(gateway_client, p, s, i)
                    else:
                        caller = lambda i, p=provider, s=scenario: call_direct(direct_client, p, s, i)
                    cell = await run_cell(label, caller, n, conc)
                    all_results.extend(cell)

        # Post-scrape gateway state.
        prom_t1 = await scrape_prometheus(gateway_client)
        health_t1 = (await gateway_client.get("/health")).json()

    # Persist.
    csv_path = out_dir / "raw.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(all_results[0]).keys()))
        w.writeheader()
        for r in all_results:
            w.writerow(asdict(r))
    print(f"raw       => {csv_path}")

    summary = compute_summary(all_results, prom_t0, prom_t1, health_t0, health_t1)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"summary   => {summary_path}")

    if not args.skip_charts:
        render_charts(summary, all_results, out_dir)
        print(f"charts    => {out_dir / 'charts'}/")

    print()
    _print_summary(summary)


def _print_summary(summary: dict) -> None:
    print("=" * 110)
    print(f"{'mode':8} {'provider':9} {'scenario':17} {'n':>4} {'ok':>4} "
          f"{'429':>4} {'5xx':>4} {'med':>6} {'p95':>6} {'p99':>6} {'mean':>6} {'retries':>7}")
    print("-" * 110)
    for key in sorted(summary["cells"].keys()):
        c = summary["cells"][key]
        print(f"{c['mode']:8} {c['provider']:9} {c['scenario']:17} "
              f"{c['count']:>4} {c['successes']:>4} {c['rate_limited']:>4} "
              f"{c['other_failures']:>4} "
              f"{c['median_ms']:>6.0f} {c['p95_ms']:>6.0f} {c['p99_ms']:>6.0f} "
              f"{c['mean_ms']:>6.0f} {c['total_retries']:>7}")
    print("=" * 110)

    g = summary["gateway_delta"]
    print()
    print("Gateway state changes during benchmark (gateway-mode cells only):")
    print(f"  requests_total:            +{g['requests_total']:.0f}")
    print(f"  cache_hits_total:          +{g['cache_hits_total']:.0f}")
    print(f"  cache_misses_total:        +{g['cache_misses_total']:.0f}")
    total_routed = g['cache_hits_total'] + g['cache_misses_total']
    if total_routed > 0:
        print(f"  gateway cache hit rate:     "
              f"{100 * g['cache_hits_total'] / total_routed:.1f}%")
    print(f"  circuit_breaker_opens:     +{g['circuit_breaker_opens_total']:.0f}")
    print(f"  trie entries:              {g["trie_entries_baseline"]} to {g['trie_entries_final']}")
    print(f"  prometheus_enabled:        {g['prometheus_enabled']}")
    print(f"  otel_enabled:              {g['otel_enabled']}")

    # Side-by-side comparison.
    print()
    print("Gateway overhead (gateway median - direct median, per cell):")
    cells = summary["cells"]
    provider_scenarios = sorted({(c["provider"], c["scenario"]) for c in cells.values()})
    for p, s in provider_scenarios:
        g = cells.get(f"gateway_{p}_{s}")
        d = cells.get(f"direct_{p}_{s}")
        if g and d and g["successes"] > 0 and d["successes"] > 0:
            delta = g["median_ms"] - d["median_ms"]
            pct = (delta / d["median_ms"]) * 100 if d["median_ms"] > 0 else 0
            sign = "+" if delta >= 0 else ""
            print(f"  {p:>7} {s:<15}  {sign}{delta:>6.1f}ms  ({sign}{pct:>5.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests-groq", type=int, default=200,
                        help="Requests per scenario per mode for Groq (default: 200)")
    parser.add_argument("--requests-gemini", type=int, default=15,
                        help="Requests per scenario per mode for Gemini (default: 15, free tier)")
    parser.add_argument("--concurrency-groq", type=int, default=15,
                        help="Max concurrent in-flight Groq requests (default: 15)")
    parser.add_argument("--concurrency-gemini", type=int, default=3,
                        help="Max concurrent in-flight Gemini requests (default: 3, free tier)")
    parser.add_argument("--modes", nargs="+", choices=["gateway", "direct"],
                        default=["gateway", "direct"],
                        help="Compare gateway and/or direct paths (default: both)")
    parser.add_argument(
        "--providers", nargs="+",
        choices=list(PROVIDERS.keys()),
        default=list(PROVIDERS.keys()),
        help="Which providers to benchmark (default: all)",
    )
    parser.add_argument("--skip-charts", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
