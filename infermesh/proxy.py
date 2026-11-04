"""HTTP proxy with per-worker circuit breaker and concurrency semaphore.

httpx errors are converted to typed exceptions so route handlers can map
them to OpenAI-format HTTP responses without catching httpx internals.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

import httpx

from infermesh import metrics
from infermesh.ratelimit import cooldown_from_headers, parse_retry_after
from infermesh.tracing import get_tracer

if TYPE_CHECKING:
    from infermesh.settings import GatewayConfig, WorkerConfig

_tracer = get_tracer(__name__)


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


class WorkerQueueFullError(WorkerUnavailableError):
    """The worker's wait queue is full or the wait timed out. Retryable on
    another worker; the client gets 503 with `retry_after_s` as Retry-After."""

    def __init__(self, worker_id: str, message: str, retry_after_s: float = 1.0) -> None:
        super().__init__(worker_id, message)
        self.retry_after_s = retry_after_s


class WorkerHTTPError(WorkerError):
    def __init__(
        self,
        worker_id: str,
        status_code: int,
        body: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(worker_id, f"upstream {status_code}")
        self.status_code = status_code
        self.body = body
        # Rate-limit hints worth passing back to the client.
        self.headers = headers or {}


def is_breaker_failure(status_code: int) -> bool:
    """True for upstream faults (5xx). 429 is a cooldown, not a fault."""
    return status_code >= 500


def retry_after_s(headers: httpx.Headers) -> float | None:
    """Parse Retry-After per RFC 9110 10.2.3: digits or an HTTP-date.

    Anything else, including "inf" and "nan", returns None so a broken header
    cannot keep a breaker open forever.
    """
    return parse_retry_after(headers.get("retry-after"))


@dataclass
class CircuitBreaker:
    failure_threshold: int = 5
    recovery_timeout_s: float = 30.0
    state: Literal["CLOSED", "OPEN", "HALF_OPEN"] = "CLOSED"
    failure_count: int = 0
    retry_at: float = 0.0
    half_open_inflight: bool = False
    max_recovery_timeout_s: float = 300.0
    consecutive_opens: int = 0

    def allow_request(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if time.monotonic() >= self.retry_at:
                self.state = "HALF_OPEN"
                self.half_open_inflight = True
                return True
            return False
        if self.half_open_inflight:
            return False
        self.half_open_inflight = True
        return True

    def is_available(self) -> bool:
        """Read-only routing check: would allow_request() admit a call now?"""
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            return time.monotonic() >= self.retry_at
        return not self.half_open_inflight

    def record_success(self) -> None:
        self.failure_count = 0
        self.consecutive_opens = 0
        self.state = "CLOSED"
        self.half_open_inflight = False

    def record_failure(self, worker_id: str = "") -> None:
        if self.state == "HALF_OPEN":
            self._open(worker_id)
            return
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold and self.state != "OPEN":
            self._open(worker_id)

    def cool_down(self, seconds: float | None, worker_id: str = "") -> None:
        """Open now for `seconds` (a 429's Retry-After), else the recovery timeout."""
        self._open(worker_id, seconds)

    def release_probe(self) -> None:
        """Free the half-open slot when a probe ends without a verdict,
        e.g. the client disconnected. Otherwise the breaker stays HALF_OPEN
        and rejects every request forever."""
        if self.state == "HALF_OPEN":
            self.half_open_inflight = False

    def _open(self, worker_id: str = "", seconds: float | None = None) -> None:
        """Open the breaker. Without an explicit cool-down the recovery time
        doubles with every consecutive ejection (capped), so a worker that keeps
        failing its probe is probed less and less often."""
        if seconds is None:
            seconds = min(
                self.recovery_timeout_s * (2 ** self.consecutive_opens),
                max(self.max_recovery_timeout_s, self.recovery_timeout_s),
            )
            self.consecutive_opens += 1
        self.state = "OPEN"
        self.retry_at = time.monotonic() + seconds
        self.half_open_inflight = False
        if worker_id:
            metrics.circuit_breaker_opens_total.labels(worker_id=worker_id).inc()


def _passthrough_headers(headers: httpx.Headers) -> dict[str, str]:
    return {
        k: v for k, v in headers.items()
        if k.lower() == "retry-after"
        or k.lower().startswith(("x-ratelimit-", "anthropic-ratelimit-"))
    }


class UpstreamStream:
    """An upstream streaming response whose status has already been checked.

    Iterate body(). aclose() is idempotent and also runs from body()'s
    finally, because Starlette skips background tasks when the iterator
    raises or (ASGI spec >= 2.4) when the client disconnects.
    """

    def __init__(
        self,
        worker: "WorkerConfig",
        breaker: CircuitBreaker,
        resp: httpx.Response,
        on_close: Callable[[], None],
    ) -> None:
        self._worker = worker
        self._breaker = breaker
        self._resp = resp
        self._on_close = on_close
        self._closed = False
        self.status_code = resp.status_code
        self.media_type = resp.headers.get("content-type", "text/event-stream")

    async def body(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._resp.aiter_bytes():
                yield chunk
        except httpx.TransportError as exc:
            self._breaker.record_failure(self._worker.id)
            raise WorkerUnavailableError(
                self._worker.id, f"stream interrupted: {exc!r}"
            ) from exc
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._resp.aclose()
        finally:
            self._on_close()


class HttpProxy:
    def __init__(self, config: "GatewayConfig") -> None:
        self._config = config
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                config.httpx_timeout_s, connect=min(5.0, config.httpx_timeout_s)
            ),
            limits=httpx.Limits(
                max_connections=config.max_concurrency_per_worker * 20,
                max_keepalive_connections=config.max_concurrency_per_worker * 10,
            ),
            follow_redirects=True,
        )
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._breakers: dict[str, CircuitBreaker] = {}
        # Requests waiting for a slot, per worker. Bounded by max_queue_per_worker.
        self._waiters: dict[str, int] = {}

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
                max_recovery_timeout_s=self._config.circuit_breaker_max_recovery_timeout_s,
            )
        return self._breakers[worker_id]

    def queued_waiters(self, worker_id: str) -> int:
        return self._waiters.get(worker_id, 0)

    def is_available(self, worker: "WorkerConfig") -> bool:
        """Routable: the breaker admits a call and the wait queue has room.

        A worker with every slot busy stays routable until `max_queue_per_worker`
        requests are waiting, so the queue is actually used before the router
        starts skipping the worker.
        """
        return (
            self._breaker(worker.id).is_available()
            and self.queued_waiters(worker.id) < self._config.max_queue_per_worker
        )

    async def _acquire_slot(self, worker_id: str) -> None:
        """Take a concurrency slot, waiting in a bounded queue.

        Raises WorkerQueueFullError when the queue is full or the wait exceeds
        `queue_timeout_s`. The waiter count is decremented in `finally`, so a
        timed-out or cancelled waiter never leaks it.
        """
        semaphore = self._semaphore(worker_id)
        if not semaphore.locked():
            await semaphore.acquire()
            return
        if self.queued_waiters(worker_id) >= self._config.max_queue_per_worker:
            metrics.queue_rejections_total.labels(worker_id=worker_id, reason="full").inc()
            raise WorkerQueueFullError(worker_id, "worker queue is full")
        self._waiters[worker_id] = self.queued_waiters(worker_id) + 1
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=self._config.queue_timeout_s)
        except asyncio.TimeoutError as exc:
            metrics.queue_rejections_total.labels(worker_id=worker_id, reason="timeout").inc()
            raise WorkerQueueFullError(
                worker_id, f"waited {self._config.queue_timeout_s:g}s for a worker slot",
                retry_after_s=self._config.queue_timeout_s,
            ) from exc
        finally:
            self._waiters[worker_id] -= 1

    def retry_in_s(self, worker: "WorkerConfig") -> float:
        """Seconds until an OPEN breaker admits a probe (0 if it would now)."""
        breaker = self._breaker(worker.id)
        if breaker.state != "OPEN":
            return 0.0
        return max(0.0, breaker.retry_at - time.monotonic())

    def _headers(self, worker: "WorkerConfig") -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if worker.api_key:
            headers["Authorization"] = f"Bearer {worker.api_key}"
        return headers

    def _request_headers(
        self,
        worker: "WorkerConfig",
        request_id: str,
        extra_headers: dict[str, str] | None,
    ) -> dict[str, str]:
        headers = self._headers(worker)
        if request_id:
            headers["X-Request-ID"] = request_id
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _span(self, worker: "WorkerConfig", path: str, request_id: str, streaming: bool):
        return _tracer.start_as_current_span(
            "worker_request",
            attributes={
                "worker.id": worker.id,
                "worker.model": worker.model,
                "request.path": path,
                "request.id": request_id,
                "request.streaming": streaming,
            },
        )

    async def _send(
        self,
        worker: "WorkerConfig",
        breaker: CircuitBreaker,
        path: str,
        body: dict,
        headers: dict[str, str],
        stream: bool,
    ) -> httpx.Response:
        """Send one request and classify the outcome for the breaker.

        Returns only 2xx/3xx responses; everything else raises a WorkerError.
        """
        url = f"{worker.url.rstrip('/')}{path}"
        try:
            if stream:
                request = self.client.build_request("POST", url, json=body, headers=headers)
                resp = await self.client.send(request, stream=True)
            else:
                resp = await self.client.post(url, json=body, headers=headers)
        except httpx.TimeoutException as exc:
            breaker.record_failure(worker.id)
            raise WorkerTimeoutError(worker.id, repr(exc)) from exc
        except httpx.TransportError as exc:
            # Connect errors, resets, "server disconnected", protocol errors.
            breaker.record_failure(worker.id)
            raise WorkerUnavailableError(worker.id, repr(exc)) from exc

        if resp.status_code < 400:
            breaker.record_success()
            return resp

        if stream:
            await resp.aread()
            await resp.aclose()
        if resp.status_code == 429:
            breaker.cool_down(
                cooldown_from_headers(resp.headers, self._config.retry_after_cap_s), worker.id
            )
        elif is_breaker_failure(resp.status_code):
            breaker.record_failure(worker.id)
        else:
            breaker.record_success()
        raise WorkerHTTPError(
            worker.id, resp.status_code, resp.text, _passthrough_headers(resp.headers)
        )

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
            try:
                headers = self._request_headers(worker, request_id, extra_headers)
                await self._acquire_slot(worker.id)
                try:
                    with self._span(worker, path, request_id, streaming=False) as span:
                        resp = await self._send(worker, breaker, path, body, headers, stream=False)
                        span.set_attribute("response.status_code", resp.status_code)
                        try:
                            return resp.json()
                        except ValueError as exc:
                            breaker.record_failure(worker.id)
                            raise WorkerUnavailableError(
                                worker.id, "upstream returned a non-JSON body"
                            ) from exc
                finally:
                    self._semaphore(worker.id).release()
            except WorkerQueueFullError:
                # No verdict on the worker itself: free a HALF_OPEN probe slot.
                breaker.release_probe()
                raise
            except WorkerError:
                raise
            except BaseException:
                breaker.release_probe()
                raise
        finally:
            if on_complete:
                on_complete(token_estimate)

    async def open_stream(
        self,
        worker: "WorkerConfig",
        path: str,
        body: dict,
        request_id: str = "",
        token_estimate: int = 0,
        on_complete: Callable[[int], None] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> UpstreamStream:
        """Open an upstream stream and return it once its status is known.

        Upstream errors raise here, before the gateway has sent a status line,
        so the client gets the real status instead of an empty 200 stream.
        The semaphore and ETIF stay held until the stream is closed.
        """
        semaphore = self._semaphore(worker.id)
        acquired = False

        def finish() -> None:
            if acquired:
                semaphore.release()
            if on_complete:
                on_complete(token_estimate)

        breaker = self._breaker(worker.id)
        if not breaker.allow_request():
            finish()
            raise WorkerCircuitOpenError(worker.id, "circuit breaker is OPEN")
        try:
            headers = self._request_headers(worker, request_id, extra_headers)
            await self._acquire_slot(worker.id)
            acquired = True
            with self._span(worker, path, request_id, streaming=True) as span:
                resp = await self._send(worker, breaker, path, body, headers, stream=True)
                span.set_attribute("response.status_code", resp.status_code)
        except WorkerQueueFullError:
            breaker.release_probe()
            finish()
            raise
        except WorkerError:
            finish()
            raise
        except BaseException:
            breaker.release_probe()
            finish()
            raise
        return UpstreamStream(worker, breaker, resp, finish)

    def breaker_states(self) -> dict[str, str]:
        return {wid: cb.state for wid, cb in self._breakers.items()}

    async def get_raw(self, worker: "WorkerConfig", path: str) -> str:
        """GET `path` at the worker's origin. vLLM serves /metrics at the root,
        not under the /v1 base URL used for completions."""
        resp = await self.client.get(
            str(httpx.URL(worker.url).join(path)),
            headers=self._headers(worker),
            timeout=5.0,
        )
        resp.raise_for_status()
        return resp.text
