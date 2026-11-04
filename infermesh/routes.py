"""FastAPI route handlers for the InferMesh gateway."""
from __future__ import annotations

import asyncio
import json
import math
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from infermesh import metrics
from infermesh.models import ErrorDetail, ErrorResponse, ModelInfo, ModelList
from infermesh.retry_budget import RetryBudget
from infermesh.proxy import (
    UpstreamStream,
    WorkerCircuitOpenError,
    WorkerError,
    WorkerHTTPError,
    WorkerQueueFullError,
    WorkerTimeoutError,
    WorkerUnavailableError,
)
from infermesh.routing import NoWorkersAvailable, _estimate_tokens
from infermesh.trie import RadixTrie
from infermesh.tracing import get_tracer

if TYPE_CHECKING:
    from infermesh.proxy import HttpProxy
    from infermesh.registry import WorkerRegistry
    from infermesh.routing import KVAwareRouter
    from infermesh.settings import GatewayConfig, WorkerConfig

log = structlog.get_logger()
_tracer = get_tracer(__name__)

router = APIRouter()


def _request_id(request: Request) -> str:
    return request.headers.get("X-Request-ID") or str(uuid.uuid4())


def _error_response(
    status: int,
    type_: str,
    message: str,
    headers: dict[str, str] | None = None,
    code: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(error=ErrorDetail(message=message, type=type_, code=code)).model_dump(),
        headers=headers,
    )


def _check_auth(request: Request) -> Response | None:
    expected = getattr(request.app.state.config, "api_key", "")
    if not expected:
        return None
    auth = request.headers.get("Authorization", "")
    if not (auth.startswith("Bearer ") and secrets.compare_digest(auth[7:], expected)):
        return JSONResponse(
            status_code=401,
            content=ErrorResponse(error=ErrorDetail(message="Invalid API key", type="invalid_request_error")).model_dump(),
            headers={"WWW-Authenticate": "Bearer"},
        )
    return None


async def _json_body(request: Request) -> dict | Response:
    try:
        body = await request.json()
    except ValueError:
        return _error_response(400, "invalid_request", "request body is not valid JSON")
    if not isinstance(body, dict):
        return _error_response(400, "invalid_request", "request body must be a JSON object")
    return body


def _worker_error_response(
    exc: WorkerError, headers: dict[str, str] | None
) -> tuple[Response, str]:
    """Map a typed proxy error to (client response, metrics status label)."""
    if isinstance(exc, WorkerCircuitOpenError):
        return _error_response(503, "circuit_open", f"worker {exc.worker_id} circuit is open", headers), "circuit_open"
    if isinstance(exc, WorkerTimeoutError):
        return _error_response(504, "timeout", f"worker {exc.worker_id} timed out", headers), "timeout"
    if isinstance(exc, WorkerQueueFullError):
        retry_in = {"Retry-After": str(max(1, math.ceil(exc.retry_after_s)))}
        return _error_response(
            503, "queue_full", f"worker {exc.worker_id} queue is full", {**retry_in, **(headers or {})}
        ), "queue_full"
    if isinstance(exc, WorkerUnavailableError):
        return _error_response(503, "worker_unavailable", f"worker {exc.worker_id} unreachable", headers), "worker_unavailable"
    assert isinstance(exc, WorkerHTTPError)
    if 400 <= exc.status_code < 500:
        # Pass 4xx through unchanged so the client sees the upstream error,
        # including Retry-After / x-ratelimit-* on 429.
        return Response(
            content=exc.body,
            status_code=exc.status_code,
            media_type="application/json",
            headers={**exc.headers, **(headers or {})},
        ), "upstream_4xx"
    return _error_response(502, "upstream_error", f"worker {exc.worker_id} returned {exc.status_code}", headers), "upstream_5xx"


def _retryable(exc: WorkerError) -> bool:
    """Safe to resend to another worker of the same model: nothing reached the
    client yet and the failure is the worker's. Timeouts are not retried; a slow
    generation would just run twice."""
    if isinstance(exc, (WorkerCircuitOpenError, WorkerUnavailableError)):
        return True
    return isinstance(exc, WorkerHTTPError) and (exc.status_code == 429 or exc.status_code >= 500)


def _hit_ratio(trie: RadixTrie, model: str, prefix: str, worker_id: str) -> float:
    """Share of the prompt `worker_id` already holds, per the prefix trie."""
    if not prefix:
        return 0.0
    _, depth = trie.longest_prefix_match(model, prefix, {worker_id})
    return min(1.0, depth / len(prefix))


def _record_usage(model: str, worker_id: str, result: dict) -> None:
    """Provider-reported prompt-cache hits: the ground truth for cache reuse."""
    usage = result.get("usage")
    if not isinstance(usage, dict):
        return
    prompt = usage.get("prompt_tokens") or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    metrics.upstream_prompt_tokens_total.labels(model=model, worker_id=worker_id).inc(prompt)
    metrics.upstream_cached_tokens_total.labels(model=model, worker_id=worker_id).inc(cached)


@dataclass
class _Route:
    worker: "WorkerConfig"
    cache_hit: bool
    label: str
    hit_source: str | None


def _select(
    request: Request,
    model: str,
    messages: list[dict] | None,
    prefix: str,
    exclude: set[str],
) -> _Route:
    kv_router: KVAwareRouter = request.app.state.router
    worker, hit = kv_router.select(model, messages or [], prefix=prefix, exclude=exclude)
    return _Route(worker, hit, "kv_aware", "trie" if hit else None)


def _sse_error(exc: WorkerError) -> bytes:
    payload = {"error": {
        "message": f"upstream stream interrupted on worker {exc.worker_id}",
        "type": "stream_error",
        "code": "upstream_stream_interrupted",
    }}
    return f"data: {json.dumps(payload)}\n\n".encode()


async def _relay(upstream: UpstreamStream, on_finish: Callable[[str], None]) -> AsyncIterator[bytes]:
    status = "success"
    try:
        async for chunk in upstream.body():
            yield chunk
    except WorkerError as exc:
        # Headers are already sent, so the status line cannot change. Tell the
        # client in-band with one SSE error event (the shape OpenAI uses for
        # mid-stream errors) instead of a silently truncated stream.
        status = "stream_error"
        log.warning("stream_error", worker_id=exc.worker_id, error=str(exc))
        yield _sse_error(exc)
    except (asyncio.CancelledError, GeneratorExit):
        status = "client_disconnect"
        raise
    finally:
        await upstream.aclose()
        on_finish(status)


async def _proxy(
    request: Request,
    *,
    endpoint: str,
    path: str,
    body: dict,
    model: str,
    prefix: str,
    messages: list[dict] | None = None,
    use_trie: bool = True,
) -> Response:
    """Route, forward, and fail over to another worker of the same model."""
    config: GatewayConfig = request.app.state.config
    proxy: HttpProxy = request.app.state.proxy
    trie: RadixTrie = request.app.state.trie
    registry: WorkerRegistry = request.app.state.registry
    req_id = _request_id(request)
    stream = bool(body.get("stream"))

    if not registry.get_workers(model):
        metrics.requests_total.labels(endpoint=endpoint, model="", status="model_not_found").inc()
        return _error_response(
            404, "invalid_request_error",
            f"model '{model}' is not served by this gateway", code="model_not_found",
        )

    token_est = _estimate_tokens(
        body, config.etif_count_output_tokens, config.etif_default_output_tokens
    )
    request_start = time.monotonic()
    tried: set[str] = set()
    last_exc: WorkerError | None = None
    budget: RetryBudget = request.app.state.retry_budget
    budget.on_request()

    for attempt in range(config.max_attempts):
        routing_start = time.monotonic()
        try:
            with _tracer.start_as_current_span("route_selection") as span:
                route = _select(request, model, messages, prefix, tried)
                span.set_attribute("worker.id", route.worker.id)
                span.set_attribute("cache.hit", route.cache_hit)
        except NoWorkersAvailable as exc:
            if last_exc is not None:
                break
            metrics.requests_total.labels(endpoint=endpoint, model=model, status="no_workers").inc()
            wait = min(proxy.retry_in_s(w) for w in registry.get_workers(model))
            headers = {"Retry-After": str(max(1, math.ceil(wait)))}
            return _error_response(503, "no_workers", str(exc), headers=headers)

        worker = route.worker
        if attempt > 0:
            # The router already picked the fallback; only now decide whether the
            # budget can afford a retry to it. Warm workers cost less credit.
            if isinstance(last_exc, (WorkerCircuitOpenError, WorkerQueueFullError)):
                # The failed attempt never left the gateway (open breaker or full
                # queue), so no upstream work was wasted: re-routing is free.
                metrics.retries_total.labels(mode="local").inc()
            else:
                hit_ratio = _hit_ratio(trie, model, prefix, worker.id) if use_trie else 0.0
                if not budget.try_acquire(hit_ratio):
                    metrics.retry_denied_total.labels(reason="budget").inc()
                    log.warning("retry_denied", request_id=req_id, worker_id=worker.id,
                                hit_ratio=round(hit_ratio, 3), credits=round(budget.credits, 3))
                    break
                metrics.retries_total.labels(mode=budget.mode).inc()
        metrics.routing_decision_seconds.labels(router=route.label).observe(time.monotonic() - routing_start)
        if use_trie:
            if route.cache_hit:
                metrics.kv_cache_hits_total.labels(model=model, source=route.hit_source or "trie").inc()
            else:
                metrics.kv_cache_misses_total.labels(model=model).inc()
        log.info("routing", request_id=req_id, model=model, worker_id=worker.id,
                 kv_cache_hit=route.cache_hit, attempt=len(tried) + 1)

        if use_trie and prefix and config.insert_on_dispatch:
            trie.insert(model, prefix, worker.id)
        worker.reserve(token_est)
        try:
            if stream:
                upstream = await proxy.open_stream(
                    worker, path, body, req_id,
                    token_estimate=token_est, on_complete=worker.release,
                )

                def on_finish(status: str, worker: WorkerConfig = worker,
                              cache_hit: bool = route.cache_hit) -> None:
                    if use_trie and prefix and status in ("success", "client_disconnect"):
                        trie.insert(model, prefix, worker.id)
                    metrics.request_duration_seconds.labels(
                        endpoint=endpoint, model=model, worker_id=worker.id,
                    ).observe(time.monotonic() - request_start)
                    metrics.requests_total.labels(endpoint=endpoint, model=model, status=status).inc()
                    log.info("request_complete", request_id=req_id, model=model,
                             worker_id=worker.id, kv_cache_hit=cache_hit,
                             stream=True, status=status)

                return StreamingResponse(
                    _relay(upstream, on_finish),
                    status_code=upstream.status_code,
                    media_type=upstream.media_type,
                    background=BackgroundTask(upstream.aclose),
                )

            result = await proxy.forward(
                worker, path, body, req_id,
                token_estimate=token_est, on_complete=worker.release,
            )
        except WorkerError as exc:
            tried.add(worker.id)
            last_exc = exc
            log.warning("worker_failed", request_id=req_id, worker_id=worker.id,
                        error=str(exc), retryable=_retryable(exc))
            if not _retryable(exc):
                break
            continue

        if use_trie and prefix:
            trie.insert(model, prefix, worker.id)
        _record_usage(model, worker.id, result)
        log.info("request_complete", request_id=req_id, model=model, worker_id=worker.id,
                 kv_cache_hit=route.cache_hit, stream=False)
        metrics.request_duration_seconds.labels(
            endpoint=endpoint, model=model, worker_id=worker.id,
        ).observe(time.monotonic() - request_start)
        metrics.requests_total.labels(endpoint=endpoint, model=model, status="success").inc()
        return JSONResponse(content=result)

    assert last_exc is not None
    response, status = _worker_error_response(last_exc, None)
    metrics.requests_total.labels(endpoint=endpoint, model=model, status=status).inc()
    return response


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    if (auth_err := _check_auth(request)) is not None:
        return auth_err
    body = await _json_body(request)
    if isinstance(body, Response):
        return body
    messages = body.get("messages", [])
    if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
        return _error_response(400, "invalid_request", "messages must be an array of objects")
    return await _proxy(
        request, endpoint="/v1/chat/completions", path="/chat/completions", body=body,
        model=body.get("model", ""), prefix=RadixTrie.extract_prefix(messages), messages=messages,
    )


@router.post("/v1/completions")
async def completions(request: Request) -> Response:
    if (auth_err := _check_auth(request)) is not None:
        return auth_err
    body = await _json_body(request)
    if isinstance(body, Response):
        return body
    prompt = body.get("prompt") if isinstance(body.get("prompt"), str) else ""
    return await _proxy(
        request, endpoint="/v1/completions", path="/completions", body=body,
        model=body.get("model", ""), prefix=prompt,
    )


@router.post("/v1/embeddings")
async def embeddings(request: Request) -> Response:
    if (auth_err := _check_auth(request)) is not None:
        return auth_err
    body = await _json_body(request)
    if isinstance(body, Response):
        return body
    return await _proxy(
        request, endpoint="/v1/embeddings", path="/embeddings", body=body,
        model=body.get("model", ""), prefix="", use_trie=False,
    )


@router.get("/v1/models")
async def list_models(request: Request) -> Response:
    if (auth_err := _check_auth(request)) is not None:
        return auth_err
    registry: WorkerRegistry = request.app.state.registry
    seen: set[str] = set()
    models: list[ModelInfo] = []
    for worker in registry.all_workers():
        if worker.model not in seen:
            seen.add(worker.model)
            models.append(ModelInfo(id=worker.model, created=int(time.time())))
    return JSONResponse(content=ModelList(data=models).model_dump())


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    registry: WorkerRegistry = request.app.state.registry
    proxy: HttpProxy = request.app.state.proxy
    trie: RadixTrie = request.app.state.trie
    workers = registry.all_workers()
    config: GatewayConfig = request.app.state.config
    breaker_states = proxy.breaker_states()

    worker_loads = [
        {
            "id": w.id,
            "model": w.model,
            "type": w.type,
            "tokens_in_flight": w.tokens_in_flight,
            "requests_in_flight": w.requests_in_flight,
            "cache_utilization": w.cache_utilization or 0.0,
            "cache_utilization_known": w.cache_utilization is not None,
            "queue_depth": w.queue_depth,
            "circuit_state": breaker_states.get(w.id, "CLOSED"),
        }
        for w in workers
    ]
    payload: dict = {
        "status": "ok",
        "workers": len(workers),
        "worker_loads": worker_loads,
        "trie_entries": trie.entry_count(),
        "cache_hits_total": int(_sum_counter(metrics.kv_cache_hits_total)),
        "redis_enabled": bool(config.redis_url),
        "redis_transport": config.redis_transport if config.redis_url else None,
        "otel_enabled": getattr(request.app.state, "otel_enabled", False),
    }

    if config.redis_url:
        payload["redis_events"] = {
            "received_total": _sum_counter(metrics.redis_events_received_total),
            "dropped_total": _sum_counter(metrics.redis_events_dropped_total),
            "by_type": _counter_by_label(metrics.redis_events_received_total, "event_type"),
            "drops_by_reason": _counter_by_label(metrics.redis_events_dropped_total, "reason"),
        }

    return JSONResponse(content=payload)


def _sum_counter(counter) -> float:
    total = 0.0
    for metric in counter.collect():
        for sample in metric.samples:
            if sample.name.endswith("_total"):
                total += sample.value
    return total


def _counter_by_label(counter, label: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in counter.collect():
        for sample in metric.samples:
            if not sample.name.endswith("_total"):
                continue
            key = sample.labels.get(label, "")
            if key:
                out[key] = out.get(key, 0.0) + sample.value
    return out


@router.get("/metrics")
async def metrics_endpoint(request: Request) -> Response:
    """Prometheus scrape. Refreshes app-state gauges on each call."""
    registry: WorkerRegistry = request.app.state.registry
    trie: RadixTrie = request.app.state.trie
    proxy: HttpProxy = request.app.state.proxy

    metrics.trie_entries.set(trie.entry_count())

    breaker_states = proxy.breaker_states()
    healthy: dict[str, int] = {}
    for w in registry.all_workers():
        metrics.worker_tokens_in_flight.labels(worker_id=w.id).set(w.tokens_in_flight)
        if w.cache_utilization is not None:
            metrics.worker_cache_utilization.labels(worker_id=w.id).set(w.cache_utilization)
        healthy.setdefault(w.model, 0)
        if breaker_states.get(w.id, "CLOSED") != "OPEN":
            healthy[w.model] += 1
    for model, count in healthy.items():
        metrics.workers_healthy.labels(model=model).set(count)

    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)
