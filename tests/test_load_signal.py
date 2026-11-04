"""Engine load signal: unknown is not idle, stale is ignored, queue depth counts, ETIF scale."""
import asyncio
import time

import httpx

from infermesh.main import poll_metrics_once
from infermesh.registry import (
    WorkerRegistry,
    parse_gpu_cache_utilization,
    parse_num_requests_waiting,
)
from infermesh.routing import PowerOfTwoChoicesRouter, _estimate_tokens
from infermesh.settings import GatewayConfig, WorkerConfig

VLLM_TEXT = (
    '# HELP vllm:num_requests_waiting Number of requests waiting to be processed.\n'
    'vllm:num_requests_running{model_name="m"} 3.0\n'
    'vllm:num_requests_waiting{model_name="m"} 5.0\n'
    'vllm:kv_cache_usage_perc{model_name="m"} 0.42\n'
)


def vllm(wid, util=None, at=None, queue=None, inflight=0):
    return WorkerConfig(id=wid, type="vllm", url="http://x", model="m", cache_utilization=util,
                        cache_utilization_at=at, queue_depth=queue, tokens_in_flight=inflight)


def test_parsers_return_none_when_the_gauge_is_absent():
    assert parse_gpu_cache_utilization(VLLM_TEXT) == 0.42
    assert parse_num_requests_waiting(VLLM_TEXT) == 5
    assert parse_gpu_cache_utilization("# nothing here\n") is None      # was 0.0 = looked idle
    assert parse_num_requests_waiting("vllm:kv_cache_usage_perc 0.1\n") is None


def test_unknown_utilization_is_scored_on_etif_alone_not_as_idle():
    p2c = PowerOfTwoChoicesRouter(etif_weight=0.6, etif_scale=1000)
    unknown = vllm("u", util=None, inflight=500)
    assert p2c._score(unknown) == 0.5                                   # ETIF only
    busy_known = vllm("k", util=0.9, at=time.monotonic(), inflight=500)
    assert p2c._score(busy_known) > p2c._score(unknown)                 # real 0.9 outweighs 0.0 guess


def test_stale_metrics_are_ignored_after_max_age():
    p2c = PowerOfTwoChoicesRouter(etif_weight=0.6, etif_scale=1000, max_age_s=30.0)
    fresh = vllm("f", util=0.9, at=time.monotonic() - 5, inflight=500)
    stale = vllm("s", util=0.9, at=time.monotonic() - 120, inflight=500)
    assert p2c._score(fresh) == 0.6 * 0.5 + 0.4 * 0.9
    assert p2c._score(stale) == 0.5                                     # last reading no longer trusted


def test_queue_depth_term_raises_the_score_and_saturates():
    p2c = PowerOfTwoChoicesRouter(etif_weight=0.4, etif_scale=1000, queue_weight=0.4, queue_scale=8)
    now = time.monotonic()
    idle = vllm("a", util=0.0, at=now, queue=0)
    backed_up = vllm("b", util=0.0, at=now, queue=4)
    flooded = vllm("c", util=0.0, at=now, queue=800)
    assert p2c._score(idle) == 0.0
    assert p2c._score(backed_up) == 0.4 * 0.5
    assert p2c._score(flooded) == 0.4                                   # clamped at queue_scale
    assert PowerOfTwoChoicesRouter(queue_weight=0.0)._score(backed_up) == 0.4 * 0.0 + 0.6 * 0.0


def test_default_scores_are_unchanged_for_known_fresh_workers():
    old = PowerOfTwoChoicesRouter(etif_weight=0.6, etif_scale=4096)
    w = vllm("w", util=0.5, at=time.monotonic(), inflight=2048)
    assert old._score(w) == 0.6 * 0.5 + 0.4 * 0.5


def test_effective_defaults():
    cfg = GatewayConfig(_env_file=None)
    assert cfg.effective_etif_scale == 20 * 1024                        # concurrency cap x 1k tokens
    assert GatewayConfig(_env_file=None, etif_scale=5000).effective_etif_scale == 5000
    assert GatewayConfig(_env_file=None, max_concurrency_per_worker=4).effective_etif_scale == 4096
    assert cfg.effective_metrics_max_age_s == 30.0                      # 3 poll intervals
    assert GatewayConfig(_env_file=None, metrics_max_age_s=7).effective_metrics_max_age_s == 7


def test_output_token_term_is_configurable():
    body = {"messages": [{"role": "user", "content": "a" * 400}], "max_tokens": 1000}
    assert _estimate_tokens(body) == 100 + 1000                         # unchanged default
    assert _estimate_tokens(body, count_output=False) == 100
    no_limit = {"messages": [{"role": "user", "content": "a" * 400}]}
    assert _estimate_tokens(no_limit) == 100 + 256
    assert _estimate_tokens(no_limit, True, 64) == 100 + 64


class FakeProxy:
    def __init__(self, replies):
        self.replies = replies

    async def get_raw(self, worker, path):
        reply = self.replies[worker.id]
        if isinstance(reply, Exception):
            raise reply
        return reply


def _registry(tmp_path):
    cfg = tmp_path / "workers.yaml"
    cfg.write_text("workers:\n  - {id: v1, type: vllm, url: 'http://v1/v1', model: m}\n"
                   "  - {id: o1, type: openai, url: 'http://o1/v1', model: m}\n")
    return WorkerRegistry(str(cfg))


def test_poll_records_values_and_timestamp(tmp_path):
    reg = _registry(tmp_path)
    before = time.monotonic()
    asyncio.run(poll_metrics_once(reg, FakeProxy({"v1": VLLM_TEXT})))   # o1 is not polled
    w = reg.all_workers()[0]
    assert (w.cache_utilization, w.queue_depth) == (0.42, 5)
    assert w.cache_utilization_at >= before


def test_failed_poll_keeps_old_values_and_their_age(tmp_path):
    reg = _registry(tmp_path)
    reg.update_metrics("v1", 0.8, 2, now=123.0)
    asyncio.run(poll_metrics_once(reg, FakeProxy({"v1": httpx.ConnectError("down")})))
    w = reg.all_workers()[0]
    assert (w.cache_utilization, w.queue_depth, w.cache_utilization_at) == (0.8, 2, 123.0)


def test_poll_without_the_gauges_does_not_overwrite_with_zero(tmp_path):
    reg = _registry(tmp_path)
    reg.update_metrics("v1", 0.8, 2, now=123.0)
    asyncio.run(poll_metrics_once(reg, FakeProxy({"v1": "# no vllm metrics\n"})))
    w = reg.all_workers()[0]
    assert (w.cache_utilization, w.cache_utilization_at) == (0.8, 123.0)
    fresh = _registry(tmp_path)                                          # never polled successfully
    asyncio.run(poll_metrics_once(fresh, FakeProxy({"v1": "# no vllm metrics\n"})))
    assert fresh.all_workers()[0].cache_utilization is None              # still unknown, not 0.0
