"""Retry credit ledger: static 10 percent rule, cache-cost rho, headroom mode, gateway wiring."""
import asyncio

import httpx
import pytest

from conftest import CHAT, asgi_client
from infermesh import metrics
from infermesh.retry_budget import RetryBudget


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _simulate(budget, requests, hit_ratio=0.0):
    """Every request fails once and asks for one retry. Returns retries admitted."""
    admitted = 0
    for _ in range(requests):
        budget.on_request()
        if budget.try_acquire(hit_ratio):
            admitted += 1
    return admitted


def test_static_mode_keeps_retries_near_ten_percent_of_requests():
    b = RetryBudget(mode="static", deposit=0.10, initial_credits=3.0, max_credits=10.0)
    admitted = _simulate(b, 1000)
    assert 100 <= admitted <= 103          # 10% of 1000 plus the 3-credit burst reserve


def test_static_mode_zero_burst_never_exceeds_deposit_rate():
    b = RetryBudget(mode="static", deposit=0.10, initial_credits=0.0)
    assert _simulate(b, 500) <= 50


def test_warm_fallback_is_cheaper_by_the_cache_cost_rule():
    cold = _simulate(RetryBudget(alpha=0.8, initial_credits=0.0, max_credits=100), 1000, hit_ratio=0.0)
    warm = _simulate(RetryBudget(alpha=0.8, initial_credits=0.0, max_credits=100), 1000, hit_ratio=1.0)
    assert cold <= 100
    assert warm >= 4.5 * cold              # rho = 0.2 vs 1.0: about 5x as many admitted


def test_rho_matches_formula_and_clamps():
    b = RetryBudget(rho_base=2.0, alpha=0.5)
    assert b.rho(0.0) == 2.0
    assert b.rho(0.6) == pytest.approx(2.0 * (1 - 0.5 * 0.6))
    assert b.rho(5.0) == pytest.approx(1.0)        # hit ratio clamped to 1
    assert b.rho(-3.0) == 2.0
    assert RetryBudget(alpha=1.0).rho(1.0) > 0     # never free


def test_credits_never_exceed_the_cap():
    b = RetryBudget(deposit=1.0, initial_credits=0.0, max_credits=5.0)
    for _ in range(100):
        b.on_request()
    assert b.credits == 5.0


def test_off_mode_admits_everything():
    b = RetryBudget(mode="off", initial_credits=0.0)
    assert all(b.try_acquire() for _ in range(50))


def test_headroom_mode_deposits_quota_over_arrival_minus_one():
    clock = Clock()
    b = RetryBudget(mode="headroom", quota_rps=100.0, window_s=10.0, initial_credits=0.0,
                    max_credits=1000.0, max_deposit=5.0, clock=clock)
    for _ in range(500):                    # 50 requests per second for 10 s
        b.on_request()
        clock.t += 0.02
    before = b.credits
    b.on_request()
    assert b.credits - before == pytest.approx(100.0 / 50.0 - 1.0, abs=0.15)   # b = Q/lambda - 1 = 1


def test_headroom_mode_admits_no_retries_when_load_exceeds_quota():
    clock = Clock()
    b = RetryBudget(mode="headroom", quota_rps=50.0, window_s=10.0, initial_credits=0.0, clock=clock)
    admitted = 0
    for _ in range(1000):                   # 100 requests per second against a 50 rps quota
        b.on_request()
        admitted += b.try_acquire()
        clock.t += 0.01
    assert admitted == 0


def _counter(name, labels):
    return metrics.REGISTRY.get_sample_value(name, labels) or 0.0


def test_gateway_denies_retry_when_budget_is_empty(make_app):
    calls = []

    def handler(request):
        calls.append(request.url.host)
        return httpx.Response(503, json={"error": {"message": "overloaded"}})

    before = _counter("infermesh_retry_denied_total", {"reason": "budget"})

    async def go():
        app = make_app(handler, workers=("w1", "w2"), retry_burst_credits=0, retry_budget_percent=0)
        async with asgi_client(app) as c:
            return await c.post("/v1/chat/completions", json=CHAT)

    r = asyncio.run(go())
    assert r.status_code == 502                       # last upstream error, no second attempt
    assert len(calls) == 1
    assert _counter("infermesh_retry_denied_total", {"reason": "budget"}) - before == 1


def test_gateway_retries_while_credits_remain(make_app):
    calls = []

    def handler(request):
        calls.append(request.url.host)
        if request.url.host == "w1":
            return httpx.Response(503, json={"error": {"message": "overloaded"}})
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}]})

    before = _counter("infermesh_retries_total", {"mode": "static"})

    async def go():
        app = make_app(handler, workers=("w1", "w2"), retry_burst_credits=3)
        app.state.trie.insert("m", "system: " + CHAT["messages"][0]["content"], "w1")   # w1 owns the prefix
        async with asgi_client(app) as c:
            return await c.post("/v1/chat/completions", json=CHAT)

    r = asyncio.run(go())
    assert r.status_code == 200
    assert calls == ["w1", "w2"]
    assert _counter("infermesh_retries_total", {"mode": "static"}) - before == 1


def test_retry_to_warm_worker_costs_less_credit(make_app):
    def handler(request):
        if request.url.host == "w1":
            return httpx.Response(503, json={"error": {"message": "overloaded"}})
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}]})

    async def go():
        app = make_app(handler, workers=("w1", "w2"), retry_burst_credits=1, retry_budget_percent=0,
                   retry_alpha=0.8)
        prefix = "\n".join(f"{m['role']}: {m['content']}" for m in CHAT["messages"])
        app.state.trie.insert("m", prefix, "w1")      # w1 is tried first
        app.state.trie.insert("m", prefix, "w2")      # w2 is warm too: hit ratio 1.0
        before = app.state.retry_budget.credits
        async with asgi_client(app) as c:
            r = await c.post("/v1/chat/completions", json=CHAT)
        return app, r, before

    app, r, before = asyncio.run(go())
    assert r.status_code == 200
    spent = before - app.state.retry_budget.credits
    assert spent == pytest.approx(0.2, abs=0.05)      # rho = 1 * (1 - 0.8 * 1.0)


def test_rerouting_after_a_local_queue_full_error_is_not_charged_to_the_budget(make_app):
    from infermesh.proxy import WorkerQueueFullError

    def handler(request):
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}]})

    before = _counter("infermesh_retries_total", {"mode": "local"})

    async def go():
        app = make_app(handler, workers=("w1", "w2"), retry_burst_credits=0, retry_budget_percent=0)
        original = app.state.proxy.forward

        async def forward(worker, *args, **kwargs):
            if worker.id == "w1":                              # lost the last slot a moment ago
                kwargs["on_complete"](kwargs["token_estimate"])
                raise WorkerQueueFullError("w1", "worker queue is full")
            return await original(worker, *args, **kwargs)

        app.state.proxy.forward = forward
        app.state.trie.insert("m", "system: " + CHAT["messages"][0]["content"], "w1")   # w1 is tried first
        async with asgi_client(app) as c:
            r = await c.post("/v1/chat/completions", json=CHAT)
        return app, r

    app, r = asyncio.run(go())
    assert r.status_code == 200                                # served by w2 with zero credits
    assert app.state.retry_budget.credits == 0.0              # nothing was withdrawn
    assert _counter("infermesh_retries_total", {"mode": "local"}) - before == 1


def test_static_mode_does_not_accumulate_arrival_history():
    b = RetryBudget(mode="static")
    for _ in range(5000):
        b.on_request()
    assert len(b._arrivals) == 0                # it used to grow by one entry per request, forever
    h = RetryBudget(mode="headroom", quota_rps=100.0, window_s=10.0, clock=Clock())
    for _ in range(5000):
        h.on_request()
    assert len(h._arrivals) == 5000             # headroom mode needs the window (trimmed when read)

