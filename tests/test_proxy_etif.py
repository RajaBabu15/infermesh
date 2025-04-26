"""HttpProxy ETIF accounting + breaker integration (no network).

Regression coverage for the token-leak bug: a circuit-open reject on the
non-streaming path must still unwind tokens_in_flight.
"""
import asyncio

import httpx

from infermesh.proxy import HttpProxy, WorkerCircuitOpenError, WorkerHTTPError
from infermesh.settings import GatewayConfig, WorkerConfig


def cfg(**kw):
    return GatewayConfig(_env_file=None, **kw)


def worker():
    return WorkerConfig(id="w", type="vllm", url="http://127.0.0.1:9", model="m")


class FakeClient:
    """Stand-in for httpx.AsyncClient.post returning a fixed status."""
    def __init__(self, status, body=None):
        self.status = status
        self.body = body or {}
        self.calls = 0

    async def post(self, url, json=None, headers=None):
        self.calls += 1
        return httpx.Response(self.status, request=httpx.Request("POST", url), json=self.body)

    async def aclose(self):
        pass


def _on_complete_factory(w):
    def _on_complete(est):
        w.tokens_in_flight = max(0, w.tokens_in_flight - est)
    return _on_complete


def test_circuit_open_reject_does_not_leak_etif():
    proxy = HttpProxy(cfg())
    proxy._breaker("w")._open("w")            # force OPEN (within recovery window)
    w = worker()
    w.tokens_in_flight = 100                  # simulate the route handler's increment

    async def go():
        try:
            await proxy.forward(w, "/chat/completions", {}, token_estimate=40,
                                on_complete=_on_complete_factory(w))
            return "ok"
        except WorkerCircuitOpenError:
            return "circuit_open"

    assert asyncio.run(go()) == "circuit_open"
    assert w.tokens_in_flight == 60           # decremented despite the open-circuit reject


def test_success_decrements_etif_exactly_once():
    proxy = HttpProxy(cfg())
    proxy.client = FakeClient(200, {"ok": True})
    w = worker()
    w.tokens_in_flight = 100

    async def go():
        return await proxy.forward(w, "/chat/completions", {}, token_estimate=40,
                                   on_complete=_on_complete_factory(w))

    assert asyncio.run(go()) == {"ok": True}
    assert w.tokens_in_flight == 60           # one decrement, not two


def test_auth_4xx_does_not_trip_breaker():
    proxy = HttpProxy(cfg(circuit_breaker_failure_threshold=2))
    proxy.client = FakeClient(401, {"error": "expired_api_key"})
    w = worker()

    async def call():
        try:
            await proxy.forward(w, "/chat/completions", {}, token_estimate=10,
                                on_complete=lambda e: None)
        except WorkerHTTPError as exc:
            return exc.status_code

    for _ in range(5):
        assert asyncio.run(call()) == 401
    assert proxy._breaker("w").state == "CLOSED"     # 401 never opens the circuit
    assert proxy._breaker("w").failure_count == 0    # counted as a healthy response


def test_server_5xx_trips_breaker():
    proxy = HttpProxy(cfg(circuit_breaker_failure_threshold=3))
    proxy.client = FakeClient(503)
    w = worker()

    async def call():
        try:
            await proxy.forward(w, "/chat/completions", {}, token_estimate=10,
                                on_complete=lambda e: None)
        except WorkerHTTPError:
            return "http_error"
        except WorkerCircuitOpenError:
            return "circuit_open"

    results = [asyncio.run(call()) for _ in range(4)]
    assert proxy._breaker("w").state == "OPEN"       # opened after 3 x 503
    assert results[-1] == "circuit_open"             # 4th request fast-fails
