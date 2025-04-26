"""HTTP proxy with per-worker circuit breaker and concurrency semaphore.

httpx errors are converted to typed exceptions so route handlers can map
them to OpenAI-format HTTP responses without catching httpx internals.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING

import httpx

from infermesh import metrics
from infermesh.tracing import get_tracer

if TYPE_CHECKING:
    from infermesh.settings import GatewayConfig, WorkerConfig

_tracer = get_tracer(__name__)


# ---------------------------------------------------------------------------
# Typed exceptions
# ---------------------------------------------------------------------------

class WorkerError(Exception):
    def __init__(self, worker_id: str, message: str) -> None:
        super().__init__(message)
        self.worker_id = worker_id


class WorkerTimeoutError(WorkerError):
    pass


class WorkerUnavailableError(WorkerError):
    pass


class WorkerCircuitOpenError(WorkerError):
    pass


class WorkerHTTPError(WorkerError):
    def __init__(self, worker_id: str, status_code: int, body: str) -> None:
        super().__init__(worker_id, f"upstream {status_code}")
        self.status_code = status_code
        self.body = body


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

def is_breaker_failure(status_code: int) -> bool:
    """Whether an upstream HTTP status reflects worker *health* (should trip
    the breaker) versus a client/config error that does not.

    5xx and 429 (overload) indicate the worker is unhealthy or shedding load.
    Other 4xx (400/401/403/404/422) mean the worker answered fine and the
    request itself was rejected — those must NOT open the circuit, otherwise an
    expired API key or a malformed request takes the whole worker offline.
    """
    return status_code >= 500 or status_code == 429


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout_s: float = 30.0
    state: Literal["CLOSED", "OPEN", "HALF_OPEN"] = "CLOSED"
    failure_count: int = 0
    opened_at: float = field(default=0.0)
    # True while a single HALF_OPEN probe is outstanding; gates concurrent
    # callers so only one request is admitted to test recovery.
    half_open_inflight: bool = False

    def allow_request(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if time.monotonic() - self.opened_at >= self.recovery_timeout_s:
                # Transition to HALF_OPEN and admit exactly one probe.
                self.state = "HALF_OPEN"
                self.half_open_inflight = True
                return True
            return False
        # HALF_OPEN: admit a single probe at a time; reject the rest.
        if self.half_open_inflight:
            return False
        self.half_open_inflight = True
        return True

    def record_success(self) -> None:
        self.failure_count = 0
        self.state = "CLOSED"
        self.half_open_inflight = False

    def record_failure(self, worker_id: str = "") -> None:
        # A failed probe in HALF_OPEN re-opens the circuit immediately.
        if self.state == "HALF_OPEN":
            self._open(worker_id)
            return
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold and self.state != "OPEN":
            self._open(worker_id)

    def _open(self, worker_id: str = "") -> None:
        self.state = "OPEN"
        self.opened_at = time.monotonic()
        self.half_open_inflight = False
        if worker_id:
            metrics.circuit_breaker_opens_total.labels(worker_id=worker_id).inc()


# ---------------------------------------------------------------------------
# HTTP proxy
# ---------------------------------------------------------------------------

class HttpProxy:
    def __init__(self, config: "GatewayConfig") -> None:
        self._config = config
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.httpx_timeout_s),
            limits=httpx.Limits(
                max_connections=config.max_concurrency_per_worker * 20,
                max_keepalive_connections=config.max_concurrency_per_worker * 10,
            ),
            follow_redirects=True,
        )
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._breakers: dict[str, CircuitBreaker] = {}

    def _semaphore(self, worker_id: str) -> asyncio.Semaphore:
        if worker_id not in self._semaphores:
            self._semaphores[worker_id] = asyncio.Semaphore(
                self._config.max_concurrency_per_worker
            )
        return self._semaphores[worker_id]

    def _breaker(self, worker_id: str) -> CircuitBreaker:
        if worker_id not in self._breakers:
            self._breakers[worker_id] = CircuitBreaker(
                failure_threshold=self._config.circuit_breaker_failure_threshold,
                recovery_timeout_s=self._config.circuit_breaker_recovery_timeout_s,
            )
        return self._breakers[worker_id]

    def _headers(self, worker: "WorkerConfig") -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if worker.api_key:
            headers["Authorization"] = f"Bearer {worker.api_key}"
        return headers

    async def forward(
        self,
        worker: "WorkerConfig",
        path: str,
        body: dict,
        request_id: str = "",
        token_estimate: int = 0,
        on_complete: Callable[[int], None] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict:
        try:
            breaker = self._breaker(worker.id)
            if not breaker.allow_request():
                raise WorkerCircuitOpenError(worker.id, "circuit breaker is OPEN")

            headers = self._headers(worker)
            if request_id:
                headers["X-Request-ID"] = request_id
            if extra_headers:
                headers.update(extra_headers)

            async with self._semaphore(worker.id):
                with _tracer.start_as_current_span(
                    "worker_request",
                    attributes={
                        "worker.id": worker.id,
                        "worker.role": worker.role,
                        "worker.model": worker.model,
                        "request.path": path,
                        "request.id": request_id,
                        "request.streaming": False,
                    },
                ) as span:
                    try:
                        resp = await self.client.post(
                            f"{worker.url.rstrip('/')}{path}",
                            json=body,
                            headers=headers,
                        )
                        resp.raise_for_status()
                        breaker.record_success()
                        span.set_attribute("response.status_code", resp.status_code)
                        return resp.json()
                    except httpx.TimeoutException as exc:
                        breaker.record_failure(worker.id)
                        span.set_attribute("error.type", "timeout")
                        raise WorkerTimeoutError(worker.id, str(exc)) from exc
                    except httpx.ConnectError as exc:
                        breaker.record_failure(worker.id)
                        span.set_attribute("error.type", "connect")
                        raise WorkerUnavailableError(worker.id, str(exc)) from exc
                    except httpx.HTTPStatusError as exc:
                        status = exc.response.status_code
                        if is_breaker_failure(status):
                            breaker.record_failure(worker.id)
                        else:
                            # Worker answered (e.g. 400/401) — it is healthy.
                            breaker.record_success()
                        span.set_attribute("error.type", "http")
                        span.set_attribute("response.status_code", status)
                        raise WorkerHTTPError(
                            worker.id, status, exc.response.text
                        ) from exc
        finally:
            # ETIF accounting must unwind on EVERY exit path — including a
            # circuit-open reject, which is raised before the request body and
            # previously leaked tokens_in_flight permanently.
            if on_complete and token_estimate:
                on_complete(token_estimate)

    async def forward_stream(
        self,
        worker: "WorkerConfig",
        path: str,
        body: dict,
        request_id: str = "",
        token_estimate: int = 0,
        on_complete: Callable[[int], None] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> AsyncIterator[bytes]:
        try:
            breaker = self._breaker(worker.id)
            if not breaker.allow_request():
                raise WorkerCircuitOpenError(worker.id, "circuit breaker is OPEN")

            headers = self._headers(worker)
            if request_id:
                headers["X-Request-ID"] = request_id
            if extra_headers:
                headers.update(extra_headers)

            async with self._semaphore(worker.id):
                with _tracer.start_as_current_span(
                    "worker_request",
                    attributes={
                        "worker.id": worker.id,
                        "worker.role": worker.role,
                        "worker.model": worker.model,
                        "request.path": path,
                        "request.id": request_id,
                        "request.streaming": True,
                    },
                ) as span:
                    try:
                        async with self.client.stream(
                            "POST",
                            f"{worker.url.rstrip('/')}{path}",
                            json=body,
                            headers=headers,
                        ) as resp:
                            resp.raise_for_status()
                            breaker.record_success()
                            span.set_attribute("response.status_code", resp.status_code)
                            async for chunk in resp.aiter_bytes():
                                yield chunk
                    except httpx.TimeoutException as exc:
                        breaker.record_failure(worker.id)
                        span.set_attribute("error.type", "timeout")
                        raise WorkerTimeoutError(worker.id, str(exc)) from exc
                    except httpx.ConnectError as exc:
                        breaker.record_failure(worker.id)
                        span.set_attribute("error.type", "connect")
                        raise WorkerUnavailableError(worker.id, str(exc)) from exc
                    except httpx.HTTPStatusError as exc:
                        status = exc.response.status_code
                        if is_breaker_failure(status):
                            breaker.record_failure(worker.id)
                        else:
                            # Worker answered (e.g. 400/401) — it is healthy.
                            breaker.record_success()
                        span.set_attribute("error.type", "http")
                        span.set_attribute("response.status_code", status)
                        raise WorkerHTTPError(
                            worker.id, status, exc.response.text
                        ) from exc
        finally:
            if on_complete and token_estimate:
                on_complete(token_estimate)

    def breaker_states(self) -> dict[str, str]:
        return {wid: cb.state for wid, cb in self._breakers.items()}

    async def get_raw(self, worker: "WorkerConfig", path: str) -> str:
        headers = self._headers(worker)
        resp = await self.client.get(
            f"{worker.url.rstrip('/')}{path}",
            headers=headers,
            timeout=5.0,
        )
        resp.raise_for_status()
        return resp.text
