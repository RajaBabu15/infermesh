"""Gateway behaviour under upstream failures, via in-process ASGI + MockTransport.

Each test pins a bug that was reproduced against a running gateway.
"""
import asyncio

import httpx

from infermesh import metrics
from infermesh.main import create_app
from infermesh.proxy import CircuitBreaker
from infermesh.registry import parse_gpu_cache_utilization

CHAT = {"model": "m", "messages": [{"role": "system", "content": "shared system prompt " * 5},
                                   {"role": "user", "content": "hi"}]}


def _app(tmp_path, monkeypatch, handler, workers=("w1",)):
    cfg = tmp_path / "workers.yaml"
    cfg.write_text("workers:\n" + "".join(
        f"  - {{id: {w}, type: openai, url: 'http://{w}/v1', model: m, api_key_env: NO_KEY}}\n"
        for w in workers))
    monkeypatch.setenv("INFERMESH_WORKERS_CONFIG_PATH", str(cfg))
    app = create_app()
    app.state.proxy.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return app


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw")


def _ok(request, content="hello"):
    return httpx.Response(200, json={
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 2,
                  "prompt_tokens_details": {"cached_tokens": 64}},
    })


def test_streaming_upstream_429_is_not_an_empty_200(tmp_path, monkeypatch):
    def handler(request):
        return httpx.Response(429, headers={"Retry-After": "7"},
                              json={"error": {"message": "slow down"}})

    async def go():
        async with _client(_app(tmp_path, monkeypatch, handler)) as c:
            return await c.post("/v1/chat/completions", json={**CHAT, "stream": True})

    r = asyncio.run(go())
    assert r.status_code == 429
    assert r.headers["retry-after"] == "7"
    assert r.json()["error"]["message"] == "slow down"


def test_streaming_success_relays_sse(tmp_path, monkeypatch):
    sse = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse)

    async def go():
        app = _app(tmp_path, monkeypatch, handler)
        async with _client(app) as c:
            r = await c.post("/v1/chat/completions", json={**CHAT, "stream": True})
        return app, r

    app, r = asyncio.run(go())
    assert r.status_code == 200 and r.content == sse
    w = app.state.registry.all_workers()[0]
    assert w.tokens_in_flight == 0 and w.requests_in_flight == 0
    assert app.state.trie.entry_count() > 0


def test_dropped_connection_is_503_not_500(tmp_path, monkeypatch):
    def handler(request):
        raise httpx.RemoteProtocolError("Server disconnected", request=request)

    async def go():
        app = _app(tmp_path, monkeypatch, handler)
        async with _client(app) as c:
            r = await c.post("/v1/chat/completions", json=CHAT)
        return app, r

    app, r = asyncio.run(go())
    assert r.status_code == 503
    assert r.json()["error"]["type"] == "worker_unavailable"
    assert app.state.proxy._breaker("w1").failure_count == 1


def test_half_open_probe_failure_does_not_wedge_breaker(tmp_path, monkeypatch):
    mode = {"fail": True}

    def handler(request):
        if mode["fail"]:
            raise httpx.RemoteProtocolError("Server disconnected", request=request)
        return _ok(request)

    async def go():
        app = _app(tmp_path, monkeypatch, handler)
        app.state.proxy._breakers["w1"] = CircuitBreaker(failure_threshold=1, recovery_timeout_s=0.0)
        async with _client(app) as c:
            first = await c.post("/v1/chat/completions", json=CHAT)    # opens the breaker
            probe = await c.post("/v1/chat/completions", json=CHAT)    # half-open probe fails
            mode["fail"] = False
            healed = await c.post("/v1/chat/completions", json=CHAT)   # upstream is back
        return app, first, probe, healed

    app, first, probe, healed = asyncio.run(go())
    assert first.status_code == 503 and probe.status_code == 503
    assert healed.status_code == 200
    assert app.state.proxy._breaker("w1").state == "CLOSED"


def test_failover_to_healthy_worker_when_owner_fails(tmp_path, monkeypatch):
    def handler(request):
        if request.url.host == "w1":
            return httpx.Response(503, json={"error": {"message": "overloaded"}})
        return _ok(request, content="from w2")

    async def go():
        app = _app(tmp_path, monkeypatch, handler, workers=("w1", "w2"))
        # w1 owns the prefix, so the router tries it first.
        app.state.trie.insert("m", "system: " + CHAT["messages"][0]["content"], "w1")
        async with _client(app) as c:
            return await c.post("/v1/chat/completions", json=CHAT)

    r = asyncio.run(go())
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "from w2"


def test_client_errors_are_not_retried(tmp_path, monkeypatch):
    calls = []

    def handler(request):
        calls.append(request.url.host)
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    async def go():
        app = _app(tmp_path, monkeypatch, handler, workers=("w1", "w2"))
        async with _client(app) as c:
            return await c.post("/v1/chat/completions", json=CHAT)

    r = asyncio.run(go())
    assert r.status_code == 400 and len(calls) == 1


def test_upstream_cached_tokens_are_recorded(tmp_path, monkeypatch):
    def sample(name):
        return metrics.REGISTRY.get_sample_value(name, {"model": "m", "worker_id": "w1"}) or 0.0

    before = sample("infermesh_upstream_cached_tokens_total"), sample("infermesh_upstream_prompt_tokens_total")

    async def go():
        async with _client(_app(tmp_path, monkeypatch, _ok)) as c:
            return await c.post("/v1/chat/completions", json=CHAT)

    assert asyncio.run(go()).status_code == 200
    assert sample("infermesh_upstream_cached_tokens_total") - before[0] == 64
    assert sample("infermesh_upstream_prompt_tokens_total") - before[1] == 100


def test_vllm_metrics_polled_at_origin_and_new_metric_name(tmp_path, monkeypatch):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, text='vllm:kv_cache_usage_perc{model_name="m"} 0.42\n')

    async def go():
        app = _app(tmp_path, monkeypatch, handler)
        w = app.state.registry.all_workers()[0]
        return await app.state.proxy.get_raw(w, "/metrics")

    text = asyncio.run(go())
    assert seen == ["http://w1/metrics"]
    assert parse_gpu_cache_utilization(text) == 0.42
    assert parse_gpu_cache_utilization('vllm:gpu_cache_usage_perc{a="b"} 0.5') == 0.5


def test_cooldown_503_carries_retry_after(tmp_path, monkeypatch):
    def handler(request):
        return httpx.Response(429, headers={"Retry-After": "20"}, json={"error": {"message": "slow"}})

    async def go():
        async with _client(_app(tmp_path, monkeypatch, handler)) as c:
            first = await c.post("/v1/chat/completions", json=CHAT)
            second = await c.post("/v1/chat/completions", json=CHAT)
        return first, second

    first, second = asyncio.run(go())
    assert first.status_code == 429 and first.headers["retry-after"] == "20"
    assert second.status_code == 503 and 1 <= int(second.headers["retry-after"]) <= 20
