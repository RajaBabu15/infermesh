"""Gateway behaviour under upstream failures, via in-process ASGI + MockTransport.

Each test pins a bug that was reproduced against a running gateway.
"""
import asyncio
import math
import time

import httpx

from conftest import CHAT, asgi_client, ok_response
from infermesh import metrics
from infermesh.proxy import CircuitBreaker
from infermesh.registry import parse_gpu_cache_utilization


def test_streaming_upstream_429_is_not_an_empty_200(make_app):
    def handler(request):
        return httpx.Response(429, headers={"Retry-After": "7"},
                              json={"error": {"message": "slow down"}})

    async def go():
        async with asgi_client(make_app(handler)) as c:
            return await c.post("/v1/chat/completions", json={**CHAT, "stream": True})

    r = asyncio.run(go())
    assert r.status_code == 429
    assert r.headers["retry-after"] == "7"
    assert r.json()["error"]["message"] == "slow down"


def test_streaming_success_relays_sse(make_app):
    sse = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse)

    async def go():
        app = make_app(handler)
        async with asgi_client(app) as c:
            r = await c.post("/v1/chat/completions", json={**CHAT, "stream": True})
        return app, r

    app, r = asyncio.run(go())
    assert r.status_code == 200 and r.content == sse
    w = app.state.registry.all_workers()[0]
    assert w.tokens_in_flight == 0 and w.requests_in_flight == 0
    assert app.state.trie.entry_count() > 0


def test_dropped_connection_is_503_not_500(make_app):
    def handler(request):
        raise httpx.RemoteProtocolError("Server disconnected", request=request)

    async def go():
        app = make_app(handler)
        async with asgi_client(app) as c:
            r = await c.post("/v1/chat/completions", json=CHAT)
        return app, r

    app, r = asyncio.run(go())
    assert r.status_code == 503
    assert r.json()["error"]["type"] == "worker_unavailable"
    assert app.state.proxy._breaker("w1").failure_count == 1


def test_half_open_probe_failure_does_not_wedge_breaker(make_app):
    mode = {"fail": True}

    def handler(request):
        if mode["fail"]:
            raise httpx.RemoteProtocolError("Server disconnected", request=request)
        return ok_response(request)

    async def go():
        app = make_app(handler)
        app.state.proxy._breakers["w1"] = CircuitBreaker(failure_threshold=1, recovery_timeout_s=0.0)
        async with asgi_client(app) as c:
            first = await c.post("/v1/chat/completions", json=CHAT)    # opens the breaker
            probe = await c.post("/v1/chat/completions", json=CHAT)    # half-open probe fails
            mode["fail"] = False
            healed = await c.post("/v1/chat/completions", json=CHAT)   # upstream is back
        return app, first, probe, healed

    app, first, probe, healed = asyncio.run(go())
    assert first.status_code == 503 and probe.status_code == 503
    assert healed.status_code == 200
    assert app.state.proxy._breaker("w1").state == "CLOSED"


def test_failover_to_healthy_worker_when_owner_fails(make_app):
    def handler(request):
        if request.url.host == "w1":
            return httpx.Response(503, json={"error": {"message": "overloaded"}})
        return ok_response(request, content="from w2")

    async def go():
        app = make_app(handler, workers=("w1", "w2"))
        # w1 owns the prefix, so the router tries it first.
        app.state.trie.insert("m", "system: " + CHAT["messages"][0]["content"], "w1")
        async with asgi_client(app) as c:
            return await c.post("/v1/chat/completions", json=CHAT)

    r = asyncio.run(go())
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "from w2"


def test_client_errors_are_not_retried(make_app):
    calls = []

    def handler(request):
        calls.append(request.url.host)
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    async def go():
        app = make_app(handler, workers=("w1", "w2"))
        async with asgi_client(app) as c:
            return await c.post("/v1/chat/completions", json=CHAT)

    r = asyncio.run(go())
    assert r.status_code == 400 and len(calls) == 1


def test_upstream_cached_tokens_are_recorded(make_app):
    def sample(name):
        return metrics.REGISTRY.get_sample_value(name, {"model": "m", "worker_id": "w1"}) or 0.0

    before = sample("infermesh_upstream_cached_tokens_total"), sample("infermesh_upstream_prompt_tokens_total")

    async def go():
        async with asgi_client(make_app(ok_response)) as c:
            return await c.post("/v1/chat/completions", json=CHAT)

    assert asyncio.run(go()).status_code == 200
    assert sample("infermesh_upstream_cached_tokens_total") - before[0] == 64
    assert sample("infermesh_upstream_prompt_tokens_total") - before[1] == 100


def test_vllm_metrics_polled_at_origin_and_new_metric_name(make_app):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, text='vllm:kv_cache_usage_perc{model_name="m"} 0.42\n')

    async def go():
        app = make_app(handler)
        w = app.state.registry.all_workers()[0]
        return await app.state.proxy.get_raw(w, "/metrics")

    text = asyncio.run(go())
    assert seen == ["http://w1/metrics"]
    assert parse_gpu_cache_utilization(text) == 0.42
    assert parse_gpu_cache_utilization('vllm:gpu_cache_usage_perc{a="b"} 0.5') == 0.5


def test_cooldown_503_carries_retry_after(make_app):
    def handler(request):
        return httpx.Response(429, headers={"Retry-After": "20"}, json={"error": {"message": "slow"}})

    async def go():
        async with asgi_client(make_app(handler)) as c:
            first = await c.post("/v1/chat/completions", json=CHAT)
            second = await c.post("/v1/chat/completions", json=CHAT)
        return first, second

    first, second = asyncio.run(go())
    assert first.status_code == 429 and first.headers["retry-after"] == "20"
    assert second.status_code == 503 and 1 <= int(second.headers["retry-after"]) <= 20


def test_429_with_only_a_reset_header_cools_the_worker_down(make_app):
    def handler(request):
        return httpx.Response(429, headers={"x-ratelimit-reset-tokens": "6m0s"}, json={"error": {"message": "slow"}})

    async def go():
        async with asgi_client(make_app(handler)) as c:
            first = await c.post("/v1/chat/completions", json=CHAT)
            second = await c.post("/v1/chat/completions", json=CHAT)
        return first, second

    first, second = asyncio.run(go())
    assert first.status_code == 429
    assert first.headers["x-ratelimit-reset-tokens"] == "6m0s"
    assert second.status_code == 503
    assert int(second.headers["retry-after"]) == 300           # 360 s hint capped at 300


def test_retry_after_inf_does_not_wedge_the_breaker(make_app):
    def handler(request):
        return httpx.Response(429, headers={"Retry-After": "inf"}, json={"error": {"message": "slow"}})

    async def go():
        app = make_app(handler)
        async with asgi_client(app) as c:
            await c.post("/v1/chat/completions", json=CHAT)
        return app

    app = asyncio.run(go())
    breaker = app.state.proxy._breaker("w1")
    assert math.isfinite(breaker.retry_at)
    assert breaker.retry_at - time.monotonic() <= 30.5         # default recovery, not forever


class _BrokenStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"par"}}]}\n\n'
        raise httpx.ReadError("connection reset")


def test_mid_stream_failure_emits_one_sse_error_event(make_app):
    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_BrokenStream())

    async def go():
        app = make_app(handler)
        async with asgi_client(app) as c:
            r = await c.post("/v1/chat/completions", json={**CHAT, "stream": True})
        return app, r

    app, r = asyncio.run(go())
    assert r.status_code == 200
    body = r.text
    assert body.startswith('data: {"choices"')
    assert body.count('"type": "stream_error"') == 1
    assert body.rstrip().endswith("}}")
    w = app.state.registry.all_workers()[0]
    assert w.tokens_in_flight == 0 and w.requests_in_flight == 0


def test_queue_wait_times_out_without_leaking_state(make_app):
    release = asyncio.Event()

    async def handler(request):
        await release.wait()
        return ok_response(request)

    async def go():
        app = make_app(handler,
                   max_concurrency_per_worker=1, max_queue_per_worker=4, queue_timeout_s=0.2)
        async with asgi_client(app) as c:
            first = asyncio.create_task(c.post("/v1/chat/completions", json=CHAT))
            await asyncio.sleep(0.05)                       # first request holds the only slot
            waiter = await c.post("/v1/chat/completions", json=CHAT)   # times out in the queue
            release.set()
            done = await first
        return app, waiter, done

    app, waiter, done = asyncio.run(go())
    assert waiter.status_code == 503
    assert waiter.json()["error"]["type"] in {"queue_full", "worker_unavailable"}
    assert int(waiter.headers["retry-after"]) >= 1
    assert done.status_code == 200
    w = app.state.registry.all_workers()[0]
    assert w.requests_in_flight == 0 and w.tokens_in_flight == 0   # no leak from the timed-out waiter
    assert app.state.proxy.queued_waiters("w1") == 0


def test_full_queue_refuses_with_retry_after_and_drains(make_app):
    release = asyncio.Event()

    async def handler(request):
        await release.wait()
        return ok_response(request)

    async def go():
        app = make_app(handler,
                   max_concurrency_per_worker=1, max_queue_per_worker=1, queue_timeout_s=5)
        async with asgi_client(app) as c:
            running = asyncio.create_task(c.post("/v1/chat/completions", json=CHAT))
            await asyncio.sleep(0.05)
            queued = asyncio.create_task(c.post("/v1/chat/completions", json=CHAT))
            await asyncio.sleep(0.05)
            refused = await c.post("/v1/chat/completions", json=CHAT)   # slot busy, queue full
            release.set()
            results = [await running, await queued]
        return app, refused, results

    app, refused, results = asyncio.run(go())
    assert refused.status_code == 503
    assert "retry-after" in refused.headers
    assert [r.status_code for r in results] == [200, 200]       # queued request was served, not dropped
    w = app.state.registry.all_workers()[0]
    assert w.requests_in_flight == 0
    assert app.state.proxy.queued_waiters("w1") == 0


def test_requests_sharing_a_prefix_are_colocated_while_the_first_is_still_in_flight(make_app):
    seen = []

    async def handler(request):
        seen.append(request.url.host)
        await asyncio.sleep(0.3)                  # the first request is still running when the next arrive
        return ok_response(request)

    async def go(flag):
        seen.clear()
        app = make_app(handler, workers=("w1", "w2", "w3", "w4"), insert_on_dispatch=flag)
        async with asgi_client(app) as c:
            first = asyncio.create_task(c.post("/v1/chat/completions", json=CHAT))
            await asyncio.sleep(0.05)
            rest = [asyncio.create_task(c.post("/v1/chat/completions", json=CHAT)) for _ in range(3)]
            await asyncio.gather(first, *rest)
        return set(seen)

    assert len(asyncio.run(go(True))) == 1        # all four went to the worker that got the first one

