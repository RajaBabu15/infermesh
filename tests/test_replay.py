"""Replay harness: trace synthesis, metric definitions, simulator mechanics.

Trace rows quoted from the Mooncake FAST25-release README (documented sample).
Simulator mechanics use tiny constructed traces: they test the model, not a claim.
"""
import json
import math
import os
from pathlib import Path

import pytest

from bench.replay.metrics import SLO, RequestResult, bootstrap_ci, meets_slo, percentile, summarize, tpot
from bench.replay.sim import Engine, ServiceModel, simulate
from bench.replay.trace import BLOCK_CHARS, TraceRequest, block_text, load_trace, synthesize_prompt
from infermesh.routing import _estimate_tokens
from infermesh.trie import RadixTrie, _common_len

# The two example rows in the Mooncake README: they share their first 12 block ids.
README_ROWS = [
    {"timestamp": 27482, "input_length": 6955, "output_length": 52,
     "hash_ids": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 2353, 2354]},
    {"timestamp": 30535, "input_length": 6472, "output_length": 26,
     "hash_ids": [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 2366]},
]


def _requests():
    return [TraceRequest(i, r["timestamp"] / 1000, r["input_length"], r["output_length"], tuple(r["hash_ids"]))
            for i, r in enumerate(README_ROWS)]


def test_prompt_length_matches_input_length_exactly_even_with_a_partial_last_block():
    for req in _requests():
        prompt = synthesize_prompt(req)
        assert len(prompt) == 4 * req.input_len                       # not a multiple of 512 * len(hash_ids)
        assert _estimate_tokens({"messages": [{"content": prompt}]}, count_output=False) == req.input_len


def test_shared_blocks_match_and_the_uncached_tail_never_does():
    a, b = (synthesize_prompt(r) for r in _requests())
    common = _common_len(a, b)
    assert common >= 12 * BLOCK_CHARS                                 # 12 shared 512-token blocks
    assert common < 12 * BLOCK_CHARS + 16                             # then the unique tails diverge
    t = RadixTrie(ttl_s=1e9)
    t.insert("m", a, "w1")
    workers, depth = t.longest_prefix_match("m", b)
    assert workers == ["w1"] and depth == common


def test_block_text_is_deterministic_and_distinct():
    assert block_text(5) == block_text(5) and len(block_text(5)) == BLOCK_CHARS
    assert block_text(5) != block_text(6)


def test_short_prompt_is_all_unique_tail():
    req = TraceRequest(0, 0.0, 100, 10, (1,))
    assert req.shared_block_ids == () and len(synthesize_prompt(req)) == 400


def test_load_trace_reads_jsonl(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in README_ROWS) + "\n")
    rows = load_trace(path)
    assert [r.input_len for r in rows] == [6955, 6472] and rows[1].ts_s == 30.535
    assert len(load_trace(path, limit=1)) == 1


def _res(ttft, e2e, out, idx=0, cached=0, inp=1000, arrival=0.0):
    return RequestResult(idx, arrival, ttft, e2e, inp, out, cached, 0)


def test_tpot_is_undefined_for_single_token_outputs():
    assert tpot(_res(0.1, 0.1, 1)) is None                            # no ZeroDivisionError
    assert tpot(_res(0.1, 0.1, 0)) is None
    assert tpot(_res(0.2, 1.2, 11)) == pytest.approx(0.1)             # (E2E - TTFT) / (out - 1)


def test_slo_check_needs_every_condition_and_skips_tpot_for_one_token():
    slo = SLO(ttft_s=0.25, tpot_s=0.10)                               # DistServe Table 1, OPT-13B chatbot
    assert meets_slo(_res(0.2, 1.2, 11), slo)                         # TTFT ok, TPOT 0.1 ok
    assert not meets_slo(_res(0.3, 0.4, 2), slo)                      # TTFT too slow
    assert not meets_slo(_res(0.2, 5.2, 11), slo)                     # TPOT 0.5 too slow
    assert meets_slo(_res(0.2, 0.2, 1), slo)                          # one-token answer: TTFT only


def test_goodput_is_slo_meeting_requests_over_duration():
    results = [_res(0.1, 1.0, 11, idx=0, arrival=0.0), _res(9.0, 10.0, 11, idx=1, arrival=0.0)]
    s = summarize(results, SLO(ttft_s=1.0, tpot_s=1.0))
    assert s["slo_attainment"] == 0.5
    assert s["goodput_rps"] == pytest.approx(1 / 10.0)                # 1 good request over a 10 s span


def test_percentile_matches_linear_interpolation():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 99) == 5
    assert math.isnan(percentile([], 50))


def test_bootstrap_ci_brackets_the_mean_and_shrinks_with_more_data():
    wide = bootstrap_ci([1.0, 2.0, 3.0, 4.0], seed=1)
    narrow = bootstrap_ci([2.5] * 40 + [2.4, 2.6], seed=1)
    assert wide[1] <= wide[0] <= wide[2]
    assert (narrow[2] - narrow[1]) < (wide[2] - wide[1])
    assert bootstrap_ci([3.0, 3.0, 3.0]) == (3.0, 3.0, 3.0)


def test_engine_lru_counts_only_consecutive_leading_blocks_and_evicts_oldest():
    e = Engine(kv_blocks=3)
    e.commit((1, 2, 3))
    assert e.cached_blocks((1, 2, 3)) == 3
    assert e.cached_blocks((9, 1, 2)) == 0                            # chain must start at the first block
    e.commit((4,))                                                    # evicts block 1
    assert e.cached_blocks((1, 2, 3)) == 0 and e.cached_blocks((2, 3, 4)) == 3


def test_second_identical_request_is_faster_because_prefill_skips_cached_blocks():
    ids = tuple(range(10))
    rows = [TraceRequest(0, 0.0, 5120, 1, ids), TraceRequest(1, 100.0, 5120, 1, ids)]
    r = simulate(rows, "round_robin", workers=1, service=ServiceModel())
    assert r[0].cached_tokens == 0 and r[1].cached_tokens == 5120
    assert r[1].ttft_s < r[0].ttft_s and r[1].ttft_s == pytest.approx(ServiceModel().prefill_s(0))


def test_fifo_prefill_queues_a_second_request_behind_the_first():
    rows = [TraceRequest(0, 0.0, 2000, 1, (1, 2, 3)), TraceRequest(1, 0.0, 2000, 1, (7, 8, 9))]
    r = simulate(rows, "round_robin", workers=1)
    prefill = ServiceModel().prefill_s(2000)
    assert r[0].ttft_s == pytest.approx(prefill)
    assert r[1].ttft_s == pytest.approx(2 * prefill)


def test_every_request_finishes_and_reservations_are_released():
    reqs = [TraceRequest(i, i * 0.5, 1500 + 10 * i, 5 + i % 3, (i % 4, 100 + i, 200 + i)) for i in range(40)]
    for policy in ("round_robin", "least_loaded", "p2c", "consistent_hash", "cache_aware",
                   "ttft_argmin", "ttft_p2c", "ttft_softmax"):
        results = simulate(reqs, policy, seed=3, workers=3, staleness_s=2.0)
        assert [r.index for r in results] == list(range(40)), policy
        assert all(r.ttft_s > 0 and r.e2e_s >= r.ttft_s for r in results), policy


def test_same_seed_same_result_and_decode_time_follows_output_length():
    reqs = [TraceRequest(i, i * 0.2, 3000, 20, (i % 3, 50 + i, 90 + i, 130 + i, 170 + i, 210 + i)) for i in range(30)]
    a = simulate(reqs, "cache_aware", seed=5, workers=3)
    b = simulate(reqs, "cache_aware", seed=5, workers=3)
    assert [(r.worker, r.ttft_s) for r in a] == [(r.worker, r.ttft_s) for r in b]
    assert all(r.e2e_s - r.ttft_s == pytest.approx(19 * ServiceModel().itl) for r in a)


def test_cache_aware_reuses_the_warm_worker_where_round_robin_cannot():
    # Same 6-block prefix, 12 requests spaced far apart on 4 workers: round robin warms
    # each worker once; the gateway's cache-aware router keeps returning to the warm one.
    ids = tuple(range(6))
    reqs = [TraceRequest(i, i * 50.0, 3072, 1, ids) for i in range(12)]
    rr = simulate(reqs, "round_robin", workers=4)
    ca = simulate(reqs, "cache_aware", seed=0, workers=4)
    assert sum(r.cached_tokens for r in ca) > sum(r.cached_tokens for r in rr)
    assert sum(1 for r in ca if r.cached_tokens > 0) == 11            # everything after the first is a hit


_TRACE_DIR = os.environ.get("MOONCAKE_TRACE_DIR")


@pytest.mark.skipif(not _TRACE_DIR, reason="set MOONCAKE_TRACE_DIR to a folder holding conversation_trace.jsonl")
def test_real_trace_rows_all_synthesize_to_the_exact_length():
    path = Path(_TRACE_DIR) / "conversation_trace.jsonl"
    assert all(len(synthesize_prompt(r)) == 4 * r.input_len for r in load_trace(path, limit=300))
    assert any(r.output_len <= 1 for r in load_trace(path))           # single-token outputs exist in the trace
