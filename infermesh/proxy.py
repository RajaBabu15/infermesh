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

@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout_s: float = 30.0
    state: Literal["CLOSED", "OPEN", "HALF_OPEN"] = "CLOSED"
    failure_count: int = 0
    opened_at: float = field(default=0.0)

    def allow_request(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if time.monotonic() - self.opened_at >= self.recovery_timeout_s:
                self.state = "HALF_OPEN"
                return True
            return False
        # HALF_OPEN: allow one probe.
        return True

    def record_success(self) -> None:
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self, worker_id: str = "") -> None:
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold and self.state != "OPEN":
            self.state = "OPEN"
            self.opened_at = time.monotonic()
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
            self._breakers[worker_id] = CircuitBreaker()
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
                    breaker.record_failure(worker.id)
                    span.set_attribute("error.type", "http")
                    span.set_attribute("response.status_code", exc.response.status_code)
                    raise WorkerHTTPError(
                        worker.id, exc.response.status_code, exc.response.text
                    ) from exc
                finally:
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
                        breaker.record_failure(worker.id)
                        span.set_attribute("error.type", "http")
                        span.set_attribute("response.status_code", exc.response.status_code)
                        raise WorkerHTTPError(
                            worker.id, exc.response.status_code, exc.response.text
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
