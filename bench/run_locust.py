"""Locust load-test orchestrator.

Spawns the gateway as a real uvicorn subprocess on a configurable port and
runs Locust headless against it. Each cell uses a fresh gateway process so
trie, circuit-breaker, and ETIF state start clean.

Output: bench/results/<utc-ts>/locust/<cell>_*.csv, _prom.json,
        manifest.json, summary.json, charts/*.png.

Usage:
  uv run python -m bench.run_locust
  uv run python -m bench.run_locust --users 10 --duration 30
  uv run python -m bench.run_locust --classes SharedPromptGroqUser
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "bench"
RESULTS_ROOT = BENCH_DIR / "results"

DEFAULT_USER_CLASSES = [
    "SharedPromptGroqUser",
    "DiverseGroqUser",
    "ConversationalGroqUser",
]


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------

def _free_port(port: int) -> None:
    try:
        out = subprocess.check_output(["lsof", "-ti", f":{port}"], text=True).strip()
        for pid in out.splitlines():
            with contextlib.suppress(Exception):
                os.kill(int(pid), signal.SIGKILL)
    except subprocess.CalledProcessError:
        pass


def _wait_for_port(port: int, timeout_s: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            try:
                s.connect(("127.0.0.1", port))
                return True
            except (OSError, socket.timeout):
                time.sleep(0.15)
    return False


def start_gateway(port: int) -> subprocess.Popen:
    _free_port(port)
    cmd = [
        sys.executable, "-m", "uvicorn",
        "infermesh.main:create_app",
        "--factory",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--log-level", "warning",
    ]
    proc = subprocess.Popen(
        cmd,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not _wait_for_port(port, timeout_s=15):
        proc.terminate()
        raise RuntimeError(f"gateway failed to bind port {port}")
    return proc


def stop_process(proc: subprocess.Popen) -> None:
    with contextlib.suppress(Exception):
        proc.terminate()
    with contextlib.suppress(Exception):
        proc.wait(timeout=3)
    if proc.poll() is None:
        with contextlib.suppress(Exception):
            proc.kill()


# ---------------------------------------------------------------------------
# Prometheus + health scrape
# ---------------------------------------------------------------------------

async def scrape_metrics(port: int) -> dict:
    """Parse infermesh_* metric families into JSON."""
    from prometheus_client.parser import text_string_to_metric_families
    async with httpx.AsyncClient(timeout=5.0) as c:
        r = await c.get(f"http://127.0.0.1:{port}/metrics")
        r.raise_for_status()
        out: dict = {}
        for fam in text_string_to_metric_families(r.text):
            if not fam.name.startswith("infermesh_"):
                continue
            out[fam.name] = [
                {"labels": dict(s.labels), "value": float(s.value)}
                for s in fam.samples
            ]
    return out


async def scrape_health(port: int) -> dict:
    async with httpx.AsyncClient(timeout=5.0) as c:
        r = await c.get(f"http://127.0.0.1:{port}/health")
        return r.json()


# ---------------------------------------------------------------------------
# Locust runner
# ---------------------------------------------------------------------------

def run_locust(
    user_class: str,
    port: int,
    users: int,
    spawn_rate: int,
    duration_s: int,
    out_dir: Path,
) -> int:
    csv_prefix = out_dir / user_class
    env = {**os.environ, "BENCH_LOCUST_CLASS": user_class}
    cmd = [
        sys.executable, "-m", "locust",
        "-f", str(BENCH_DIR / "locustfile_real.py"),
        "--headless",
        "--users", str(users),
        "--spawn-rate", str(spawn_rate),
        "--run-time", f"{duration_s}s",
        "--host", f"http://127.0.0.1:{port}",
        "--csv", str(csv_prefix),
        "--only-summary",
        "--loglevel", "WARNING",
    ]
    return subprocess.call(cmd, env=env)


# ---------------------------------------------------------------------------
# Per-cell runner
# ---------------------------------------------------------------------------

async def run_cell(
    user_class: str,
    port: int,
    users: int,
    spawn_rate: int,
    duration_s: int,
    out_dir: Path,
) -> None:
    print(f"\n=== cell: {user_class}  (users={users}, duration={duration_s}s) ===")
    gateway = start_gateway(port)
    try:
        await asyncio.sleep(1.5)  # lifespan tasks settle
        prom_t0 = await scrape_metrics(port)
        health_t0 = await scrape_health(port)
        print(f"  gateway up on port {port}, running locust...")

        rc = run_locust(user_class, port, users, spawn_rate, duration_s, out_dir)
        if rc != 0:
            print(f"  locust exited with code {rc}")

        prom_t1 = await scrape_metrics(port)
        health_t1 = await scrape_health(port)

        snapshot_path = out_dir / f"{user_class}_prom.json"
        snapshot_path.write_text(json.dumps({
            "user_class": user_class,
            "users": users,
            "spawn_rate": spawn_rate,
            "duration_s": duration_s,
            "prom_t0": prom_t0,
            "prom_t1": prom_t1,
            "health_t0": health_t0,
            "health_t1": health_t1,
        }, indent=2))
    finally:
        stop_process(gateway)
        await asyncio.sleep(1.0)  # let port release


# ---------------------------------------------------------------------------
# Analysis (deltas + chart)
# ---------------------------------------------------------------------------

def _sum_counter(metrics: dict, name: str) -> float:
    samples = metrics.get(name) or metrics.get(name.removesuffix("_total"), [])
    return sum(s["value"] for s in samples)


def _by_label(metrics: dict, name: str, label: str) -> dict[str, float]:
    samples = metrics.get(name) or metrics.get(name.removesuffix("_total"), [])
    out: dict[str, float] = {}
    for s in samples:
        key = s["labels"].get(label, "")
        if key:
            out[key] = out.get(key, 0.0) + s["value"]
    return out


def build_summary(out_dir: Path, manifest: dict) -> dict:
    import pandas as pd  # noqa
    cells: dict[str, dict] = {}
    for cls in manifest["classes"]:
        prom_path = out_dir / f"{cls}_prom.json"
        stats_path = out_dir / f"{cls}_stats.csv"
        if not prom_path.exists() or not stats_path.exists():
            print(f"  skip {cls}: missing data")
            continue
        prom = json.loads(prom_path.read_text())
        df = pd.read_csv(stats_path)
        agg = df[df["Name"] == "Aggregated"].iloc[0]

        delta_hits = _sum_counter(prom["prom_t1"], "infermesh_kv_cache_hits_total") \
                   - _sum_counter(prom["prom_t0"], "infermesh_kv_cache_hits_total")
        delta_miss = _sum_counter(prom["prom_t1"], "infermesh_kv_cache_misses_total") \
                   - _sum_counter(prom["prom_t0"], "infermesh_kv_cache_misses_total")
        delta_cb = _sum_counter(prom["prom_t1"], "infermesh_circuit_breaker_opens_total") \
                 - _sum_counter(prom["prom_t0"], "infermesh_circuit_breaker_opens_total")
        delta_total = _sum_counter(prom["prom_t1"], "infermesh_requests_total") \
                    - _sum_counter(prom["prom_t0"], "infermesh_requests_total")
        delta_by_status = {
            label: _by_label(prom["prom_t1"], "infermesh_requests_total", "status").get(label, 0)
                 - _by_label(prom["prom_t0"], "infermesh_requests_total", "status").get(label, 0)
            for label in ("success", "circuit_open", "upstream_4xx", "upstream_5xx",
                          "timeout", "worker_unavailable", "stream_error")
        }
        hit_rate = (delta_hits / (delta_hits + delta_miss) * 100) if (delta_hits + delta_miss) > 0 else 0.0

        cells[cls] = {
            "requests": int(agg["Request Count"]),
            "failures": int(agg["Failure Count"]),
            "rps": float(agg["Requests/s"]),
            "median_ms": float(agg["Median Response Time"]),
            "p95_ms": float(agg["95%"]),
            "p99_ms": float(agg["99%"]),
            "mean_ms": float(agg["Average Response Time"]),
            "gateway_routed": int(delta_total),
            "gateway_cache_hits": int(delta_hits),
            "gateway_cache_misses": int(delta_miss),
            "gateway_cache_hit_rate_pct": round(hit_rate, 1),
            "circuit_breaker_opens": int(delta_cb),
            "by_status": delta_by_status,
        }
    return cells


def render_charts(summary: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts_dir = out_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    classes = list(summary.keys())
    if not classes:
        return

    # Chart 1: p99 latency per cell.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    p99 = [summary[c]["p99_ms"] for c in classes]
    bars = ax.bar(classes, p99, color="#2ca02c", edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, p99):
        ax.text(bar.get_x() + bar.get_width() / 2, val,
                f"{val:.0f}ms", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("p99 latency (ms)")
    ax.set_title("Locust p99 latency by user class")
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(charts_dir / "p99_by_class.png", dpi=120)
    plt.close(fig)

    # Chart 2: gateway cache-hit rate per cell.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    rates = [summary[c]["gateway_cache_hit_rate_pct"] for c in classes]
    bars = ax.bar(classes, rates, color="#1f77b4", edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, val,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("gateway cache hit rate (%)")
    ax.set_title("Gateway trie hit rate by user class")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(charts_dir / "cache_hit_rate_by_class.png", dpi=120)
    plt.close(fig)

    # Chart 3: stacked outcome counts by gateway status.
    statuses = ["success", "circuit_open", "upstream_4xx", "upstream_5xx",
                "timeout", "worker_unavailable", "stream_error"]
    status_colors = {
        "success": "#2ca02c",
        "circuit_open": "#d62728",
        "upstream_4xx": "#ff7f0e",
        "upstream_5xx": "#8c564b",
        "timeout": "#9467bd",
        "worker_unavailable": "#7f7f7f",
        "stream_error": "#e377c2",
    }
    fig, ax = plt.subplots(figsize=(10, 5))
    x = list(range(len(classes)))
    bottoms = [0.0] * len(classes)
    for status in statuses:
        vals = [summary[c]["by_status"].get(status, 0) for c in classes]
        if all(v == 0 for v in vals):
            continue
        ax.bar(x, vals, bottom=bottoms, label=status, color=status_colors[status],
               edgecolor="black", linewidth=0.5)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=9)
    ax.set_ylabel("gateway requests (by status label)")
    ax.set_title("Per-cell gateway request outcomes")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(charts_dir / "outcomes_by_class.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _summary_table(summary: dict) -> None:
    print()
    print("=" * 110)
    print(f"{'user_class':30} {'reqs':>5} {'rps':>6} {'med':>6} {'p95':>6} {'p99':>6} "
          f"{'hits':>5} {'miss':>5} {'hit%':>5} {'cb':>4}")
    print("-" * 110)
    for cls, c in summary.items():
        print(f"{cls:30} {c['requests']:>5} {c['rps']:>6.1f} "
              f"{c['median_ms']:>6.0f} {c['p95_ms']:>6.0f} {c['p99_ms']:>6.0f} "
              f"{c['gateway_cache_hits']:>5} {c['gateway_cache_misses']:>5} "
              f"{c['gateway_cache_hit_rate_pct']:>5.0f} "
              f"{c['circuit_breaker_opens']:>4}")
    print("=" * 110)


async def main_async(args: argparse.Namespace) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = RESULTS_ROOT / ts / "locust"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"InferMesh Locust load test, {ts}")
    print(f"  output:   {out_dir}")
    print(f"  classes:  {', '.join(args.classes)}")
    print(f"  users:    {args.users}  spawn={args.spawn_rate}/s  "
          f"duration={args.duration}s  gateway-port={args.port}")

    for cls in args.classes:
        await run_cell(cls, args.port, args.users, args.spawn_rate,
                       args.duration, out_dir)

    manifest = {
        "timestamp": ts,
        "classes": args.classes,
        "users": args.users,
        "spawn_rate": args.spawn_rate,
        "duration_s": args.duration,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    summary = build_summary(out_dir, manifest)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    render_charts(summary, out_dir)
    print(f"\nresults: {out_dir}")
    print(f"charts:  {out_dir / 'charts'}/")
    _summary_table(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=20,
                        help="Peak concurrent Locust users (default: 20)")
    parser.add_argument("--spawn-rate", type=int, default=5,
                        help="User ramp-up rate per second (default: 5)")
    parser.add_argument("--duration", type=int, default=45,
                        help="Cell runtime in seconds (default: 45)")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_USER_CLASSES,
                        choices=DEFAULT_USER_CLASSES)
    parser.add_argument("--port", type=int, default=8765,
                        help="Port for the spawned gateway (default: 8765)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
