"""E1 sweep with 20 seeds and 95 percent bootstrap intervals.

  uv run python -m bench.retry.run

Part A (Bronson loop, one pool of capacity 300 rps, 10 s outage): fixed two attempts vs
no retries vs the static 10 percent budget vs the headroom budget, at several loads.
Part B (cache-cost link): four servers with prefix affinity, warm attempts cost 20 percent
of cold ones, and the outage hits server 0 only, so retries fail over to servers 1 to 3,
which then hold the prefix warm. Budget with alpha = 0 (every retry costs 1) vs alpha = 0.8.
The window metrics cover requests that arrive during the outage.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from bench.replay.metrics import bootstrap_ci
from bench.retry.sim import Params, Policy, run

METRICS = ["steady_goodput_rps", "window_success_ratio", "window_attempts_per_request", "recovered"]


def _cell(params, policy, seeds):
    rows = [run(params, policy, s) for s in range(seeds)]
    out = {}
    for m in METRICS:
        mean, lo, hi = bootstrap_ci([r[m] for r in rows])
        out[m] = {"mean": mean, "ci95": [lo, hi]}
    out["seeds"] = seeds
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cells: dict[str, dict] = {}
    for load in (210.0, 280.0, 295.0):
        params = Params(load_rps=load, duration_s=args.duration)
        for pol in (Policy("none", "none"), Policy("fixed2", "fixed"),
                    Policy("static10", "static"), Policy("headroom", "headroom")):
            cells[f"A load={load:g} {pol.name}"] = _cell(params, pol, args.seeds)

    params_b = Params(servers=4, load_rps=150.0, duration_s=args.duration, warm_saving=0.8,
                      prefixes=8, outage_servers=(0,))
    for mode in ("static", "headroom"):
        for alpha in (0.0, 0.8):
            pol = Policy(f"{mode} alpha={alpha:g}", mode, alpha=alpha)
            cells[f"B load=150 {pol.name}"] = _cell(params_b, pol, args.seeds)
    for name, mode in (("none", "none"), ("fixed2", "fixed")):
        cells[f"B load=150 {name}"] = _cell(params_b, Policy(name, mode), args.seeds)

    out_dir = Path(args.out or f"bench/results/retry-{date.today().isoformat()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({
        "model": "Bronson HotOS 2021 Sec 2.1 loop: capacity 300 rps, timeout 1 s, 10 s outage",
        "params_a": vars(Params(duration_s=args.duration)), "params_b": vars(params_b),
        "cells": cells}, indent=2))
    lines = ["| cell | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for cell, agg in cells.items():
        cols = [f"{agg[m]['mean']:.3f} [{agg[m]['ci95'][0]:.3f}, {agg[m]['ci95'][1]:.3f}]" for m in METRICS]
        lines.append(f"| {cell} | " + " | ".join(cols) + " |")
    table = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(table)
    print(table)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
