"""Run the replay and write results with bootstrap confidence intervals.

  uv run python -m bench.replay.run --trace PATH/conversation_trace.jsonl \
      --experiment policies --limit 3000 --workers 8 --seeds 20

Experiments
  policies   round_robin, least_loaded, p2c, consistent_hash, cache_aware (live load)
  staleness  ttft_argmin / ttft_p2c / ttft_softmax with the engine backlog snapshot
             T seconds old, for T in --stale-values (E2, a hypothesis test)

Deterministic policies run once; stochastic ones run once per seed. Each metric is
reported as mean and 95 percent percentile-bootstrap interval over seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import date
from pathlib import Path

from bench.replay.metrics import SLO, bootstrap_ci, summarize
from bench.replay.sim import POLICIES, ServiceModel, simulate
from bench.replay.trace import load_trace

METRICS = ["ttft_p50_s", "ttft_p99_s", "slo_attainment", "goodput_rps", "cached_token_ratio", "e2e_p99_s"]


def _is_stochastic(name: str) -> bool:
    return POLICIES[name]().stochastic


def _run_cell(requests, policy, seeds, slo, **sim_kwargs):
    seeds = seeds if _is_stochastic(policy) else [0]
    rows = []
    for seed in seeds:
        started = time.time()
        summary = summarize(simulate(requests, policy, seed=seed, **sim_kwargs), slo)
        summary.update(policy=policy, seed=seed, wall_s=round(time.time() - started, 2))
        rows.append(summary)
    return rows


def _aggregate(rows):
    out = {}
    for metric in METRICS:
        mean, low, high = bootstrap_ci([r[metric] for r in rows])
        out[metric] = {"mean": mean, "ci95": [low, high]}
    out["seeds"] = len(rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--experiment", choices=["policies", "staleness"], default="policies")
    ap.add_argument("--limit", type=int, default=3000, help="first N requests of the trace")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--kv-blocks", type=int, default=4000, help="per-worker KV cache, in 512-token blocks")
    ap.add_argument("--speedup", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--slo-ttft", type=float, default=2.0,
                    help="free parameter; DistServe's 0.25 s chatbot SLO is for short prompts")
    ap.add_argument("--slo-tpot", type=float, default=0.10)
    ap.add_argument("--stale-values", type=float, nargs="+", default=[0.0, 5.0, 20.0])
    ap.add_argument("--softmax-c", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    requests = load_trace(args.trace, args.limit)
    slo = SLO(args.slo_ttft, args.slo_tpot)
    seeds = list(range(args.seeds))
    base = dict(workers=args.workers, kv_blocks=args.kv_blocks, speedup=args.speedup, service=ServiceModel())

    cells: dict[str, list[dict]] = {}
    if args.experiment == "policies":
        for policy in ["round_robin", "least_loaded", "p2c", "consistent_hash", "cache_aware"]:
            cells[policy] = _run_cell(requests, policy, seeds, slo, **base)
    else:
        for stale in args.stale_values:
            for policy in ["ttft_argmin", "ttft_p2c", "ttft_softmax"]:
                cells[f"{policy}@T={stale:g}"] = _run_cell(
                    requests, policy, seeds, slo, staleness_s=stale, softmax_c=args.softmax_c, **base)

    out_dir = Path(args.out or f"bench/results/replay-{args.experiment}-{date.today().isoformat()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = [dict(r, cell=cell) for cell, rows in cells.items() for r in rows]
    with open(out_dir / "runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in all_rows for k in r}), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "experiment": args.experiment,
        "trace": Path(args.trace).name,
        "requests": len(requests),
        "config": {"workers": args.workers, "kv_blocks": args.kv_blocks, "speedup": args.speedup,
                   "seeds": args.seeds, "slo": {"ttft_s": slo.ttft_s, "tpot_s": slo.tpot_s},
                   "service_model": vars(ServiceModel()) if hasattr(ServiceModel(), "__dict__") else {}},
        "cells": {cell: _aggregate(rows) for cell, rows in cells.items()},
        "note": "Service-time constants are assumptions (DistServe publishes the form, not the values): "
                "read the ordering of policies, not the absolute latencies.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    header = f"| cell | seeds | " + " | ".join(METRICS) + " |\n|---|---|" + "---|" * len(METRICS) + "\n"
    lines = []
    for cell, agg in summary["cells"].items():
        cols = []
        for m in METRICS:
            mean, (lo, hi) = agg[m]["mean"], agg[m]["ci95"]
            cols.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}]")
        lines.append(f"| {cell} | {agg['seeds']} | " + " | ".join(cols) + " |")
    table = header + "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(table)
    print(table)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
