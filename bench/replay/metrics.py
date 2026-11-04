"""Latency and goodput metrics for the replay.

Definitions follow vLLM's benchmark and the DistServe paper:
  TTFT = time from arrival to the first output token
  E2E  = time from arrival to the last output token
  TPOT = (E2E - TTFT) / (output_len - 1), defined only when output_len > 1
         (a one-token answer has no inter-token interval)
  goodput = requests that meet EVERY SLO / duration
DistServe Table 1, OPT-13B chatbot on ShareGPT: TTFT 0.25 s, TPOT 0.1 s.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SLO:
    ttft_s: float = 0.25
    tpot_s: float = 0.10


@dataclass(frozen=True)
class RequestResult:
    index: int
    arrival_s: float
    ttft_s: float
    e2e_s: float
    input_len: int
    output_len: int
    cached_tokens: int               # prompt tokens the engine served from its KV cache
    worker: int


def tpot(result: RequestResult) -> float | None:
    if result.output_len <= 1:
        return None
    return (result.e2e_s - result.ttft_s) / (result.output_len - 1)


def meets_slo(result: RequestResult, slo: SLO) -> bool:
    if result.ttft_s > slo.ttft_s:
        return False
    t = tpot(result)
    return t is None or t <= slo.tpot_s


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in 0..100), the numpy default method."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def summarize(results: list[RequestResult], slo: SLO = SLO()) -> dict[str, float]:
    if not results:
        return {}
    duration = max(r.arrival_s + r.e2e_s for r in results) - min(r.arrival_s for r in results)
    ttft = [r.ttft_s for r in results]
    tpots = [t for t in (tpot(r) for r in results) if t is not None]
    good = sum(meets_slo(r, slo) for r in results)
    return {
        "requests": float(len(results)),
        "ttft_p50_s": percentile(ttft, 50),
        "ttft_p99_s": percentile(ttft, 99),
        "tpot_p50_s": percentile(tpots, 50),
        "tpot_p99_s": percentile(tpots, 99),
        "e2e_p50_s": percentile([r.e2e_s for r in results], 50),
        "e2e_p99_s": percentile([r.e2e_s for r in results], 99),
        "slo_attainment": good / len(results),
        "goodput_rps": good / duration if duration > 0 else float("nan"),
        "cached_token_ratio": sum(r.cached_tokens for r in results) / max(1, sum(r.input_len for r in results)),
    }


def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """(mean, low, high): percentile bootstrap of the mean over per-seed values."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return (sum(values) / n,
            means[int(n_boot * alpha / 2)],
            means[min(n_boot - 1, int(n_boot * (1 - alpha / 2)))])
