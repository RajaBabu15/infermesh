"""FastAPI route handlers for the InferMesh gateway."""
from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

import structlog
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from infermesh import metrics
from infermesh.models import ErrorDetail, ErrorResponse, ModelInfo, ModelList
from infermesh.proxy import (
    WorkerCircuitOpenError,
    WorkerHTTPError,
    WorkerTimeoutError,
    WorkerUnavailableError,
)
from infermesh.routing import NoWorkersAvailable, RadixTrie, _estimate_tokens
from infermesh.tracing import get_tracer

if TYPE_CHECKING:
    from infermesh.proxy import HttpProxy
    from infermesh.registry import WorkerRegistry
    from infermesh.routing import DisaggregatedRouter, KVAwareRouter, SessionTracker
    from infermesh.settings import GatewayConfig

log = structlog.get_logger()
_tracer = get_tracer(__name__)

router = APIRouter()


def _request_id(request: Request) -> str:
    return request.headers.get("X-Request-ID") or str(uuid.uuid4())


def _error_response(
    status: int, type_: str, message: str, headers: dict[str, str] | None = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(error=ErrorDetail(message=message, type=type_)).model_dump(),
        headers=headers,
    )


def _check_auth(request: Request) -> Response | None:
    expected = getattr(request.app.state.config, "api_key", "")
    if not expected:
        return None
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer ") or auth[7:] != expected:
        return JSONResponse(
            status_code=401,
            content=ErrorResponse(error=ErrorDetail(message="Invalid API key", type="invalid_request_error")).model_dump(),
            headers={"WWW-Authenticate": "Bearer"},
        )
    return None


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err
    req_id = _request_id(request)
    try:
        body = await request.json()
    except Exception:
        return _error_response(400, "invalid_request", "request body is not valid JSON")
    model = body.get("model", "")
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    config: GatewayConfig = request.app.state.config
    proxy: HttpProxy = request.app.state.proxy
    trie: RadixTrie = request.app.state.trie

    response_sid: str | None = None
    upstream_headers: dict[str, str] = {}

    routing_start = time.monotonic()
    request_start = routing_start
    try:
        with _tracer.start_as_current_span("route_selection") as route_span:
            if config.disaggregation_enabled:
                disagg_router: DisaggregatedRouter = request.app.state.disagg_router
                incoming_sid = request.headers.get(config.session_header_name)

                if incoming_sid:
                    # Sticky-route to the decode worker holding this KV.
                    worker, cache_hit = disagg_router.select_decode(
                        model, incoming_sid, messages
                    )
                    response_sid = incoming_sid
                    router_label = "disaggregated_decode"
                    hit_source = "session_sticky" if cache_hit else None
                else:
                    # New session: route to prefill and mint a session_id.
                    worker, cache_hit = disagg_router.select_prefill(model, messages)
                    response_sid = f"sess-{uuid.uuid4().hex[:16]}"
                    router_label = "disaggregated_prefill"
                    hit_source = "trie" if cache_hit else None

                upstream_headers[config.session_header_name] = response_sid
                route_span.set_attribute("disagg.enabled", True)
                route_span.set_attribute("session.id", response_sid)
            else:
                kv_router: KVAwareRouter = request.app.state.router
                worker, cache_hit = kv_router.select(model, messages)
                router_label = "kv_aware"
                hit_source = "trie" if cache_hit else None
                route_span.set_attribute("disagg.enabled", False)

            route_span.set_attribute("worker.id", worker.id)
            route_span.set_attribute("worker.role", worker.role)
            route_span.set_attribute("cache.hit", cache_hit)
    except NoWorkersAvailable as exc:
        metrics.requests_total.labels(
            endpoint="/v1/chat/completions", model=model, status="no_workers"
        ).inc()
        return _error_response(503, "no_workers", str(exc))

    metrics.routing_decision_seconds.labels(router=router_label).observe(
        time.monotonic() - routing_start
    )
    if cache_hit:
        metrics.kv_cache_hits_total.labels(model=model, source=hit_source or "trie").inc()
    else:
        metrics.kv_cache_misses_total.labels(model=model).inc()

    log.info(
        "routing",
        request_id=req_id,
        model=model,
        worker_id=worker.id,
        kv_cache_hit=cache_hit,
        session_id=response_sid,
    )

    token_est = _estimate_tokens(body)
    worker.reserve(token_est)

    def _on_complete(est: int) -> None:
        worker.release(est)

    # Session header is returned on every response (incl. errors) so a
    # conversational client keeps its sticky session across upstream failures.
    sess_hdr = {config.session_header_name: response_sid} if response_sid else None

    try:
        if stream:
            prefix = RadixTrie.extract_prefix(messages)

            async def _stream():
                stream_status = "success"
                try:
                    async for chunk in proxy.forward_stream(
                        worker, "/chat/completions", body, req_id,
                        token_estimate=token_est,
                        on_complete=_on_complete,
                        extra_headers=upstream_headers or None,
                    ):
                        yield chunk
                    trie.insert(model, prefix, worker.id)
                    request.app.state.cache_hits_total += int(cache_hit)
                    log.info(
                        "request_complete",
                        request_id=req_id,
                        model=model,
                        worker_id=worker.id,
                        kv_cache_hit=cache_hit,
                        stream=True,
                    )
                except Exception as exc:
                    stream_status = "stream_error"
                    log.warning(
                        "stream_error",
                        request_id=req_id,
                        worker_id=worker.id,
                        error=str(exc),
                    )
                finally:
                    metrics.request_duration_seconds.labels(
                        endpoint="/v1/chat/completions",
                        model=model,
                        worker_id=worker.id,
                    ).observe(time.monotonic() - request_start)
                    metrics.requests_total.labels(
                        endpoint="/v1/chat/completions",
                        model=model,
                        status=stream_status,
                    ).inc()

            stream_headers = (
                {config.session_header_name: response_sid} if response_sid else None
            )
            return StreamingResponse(
                _stream(),
                media_type="text/event-stream",
                headers=stream_headers,
            )

        result = await proxy.forward(
            worker, "/chat/completions", body, req_id,
            token_estimate=token_est,
            on_complete=_on_complete,
            extra_headers=upstream_headers or None,
        )
        prefix = RadixTrie.extract_prefix(messages)
        trie.insert(model, prefix, worker.id)
        request.app.state.cache_hits_total += int(cache_hit)
        log.info(
            "request_complete",
            request_id=req_id,
            model=model,
            worker_id=worker.id,
            kv_cache_hit=cache_hit,
            stream=False,
            tokens_in_flight=worker.tokens_in_flight,
        )
        metrics.request_duration_seconds.labels(
            endpoint="/v1/chat/completions", model=model, worker_id=worker.id,
        ).observe(time.monotonic() - request_start)
        metrics.requests_total.labels(
            endpoint="/v1/chat/completions", model=model, status="success",
        ).inc()
        response_headers = (
            {config.session_header_name: response_sid} if response_sid else None
        )
        return JSONResponse(content=result, headers=response_headers)

    except WorkerCircuitOpenError as exc:
        metrics.requests_total.labels(
            endpoint="/v1/chat/completions", model=model, status="circuit_open",
        ).inc()
        return _error_response(503, "circuit_open", f"worker {exc.worker_id} circuit is open", headers=sess_hdr)
    except WorkerTimeoutError as exc:
        metrics.requests_total.labels(
            endpoint="/v1/chat/completions", model=model, status="timeout",
        ).inc()
        return _error_response(504, "timeout", f"worker {exc.worker_id} timed out", headers=sess_hdr)
    except WorkerUnavailableError as exc:
        metrics.requests_total.labels(
            endpoint="/v1/chat/completions", model=model, status="worker_unavailable",
        ).inc()
        return _error_response(503, "worker_unavailable", f"worker {exc.worker_id} unreachable", headers=sess_hdr)
    except WorkerHTTPError as exc:
        status = "upstream_4xx" if 400 <= exc.status_code < 500 else "upstream_5xx"
        metrics.requests_total.labels(
            endpoint="/v1/chat/completions", model=model, status=status,
        ).inc()
        # Pass 4xx through unchanged so the client sees the upstream error.
        if 400 <= exc.status_code < 500:
            return Response(content=exc.body, status_code=exc.status_code, media_type="application/json", headers=sess_hdr)
        return _error_response(502, "upstream_error", f"worker {exc.worker_id} returned {exc.status_code}", headers=sess_hdr)


@router.post("/v1/completions")
async def completions(request: Request) -> Response:
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err
    req_id = _request_id(request)
    try:
        body = await request.json()
    except Exception:
        return _error_response(400, "invalid_request", "request body is not valid JSON")
    model = body.get("model", "")
    stream = body.get("stream", False)
    prompt = body.get("prompt", "") if isinstance(body.get("prompt"), str) else ""

    kv_router: KVAwareRouter = request.app.state.router
    proxy: HttpProxy = request.app.state.proxy
    trie: RadixTrie = request.app.state.trie

    try:
        worker, cache_hit = kv_router.select(model, [], prefix=prompt)
    except NoWorkersAvailable as exc:
        return _error_response(503, "no_workers", str(exc))

    log.info("routing", request_id=req_id, model=model, worker_id=worker.id, kv_cache_hit=cache_hit)

    token_est = _estimate_tokens(body)
    worker.reserve(token_est)

    def _on_complete(est: int) -> None:
        worker.release(est)

    try:
        if stream:
            async def _stream():
                try:
                    async for chunk in proxy.forward_stream(
                        worker, "/completions", body, req_id,
                        token_estimate=token_est,
                        on_complete=_on_complete,
                    ):
                        yield chunk
                    if prompt:
                        trie.insert(model, prompt, worker.id)
                except Exception as exc:
                    log.warning("stream_error", request_id=req_id, worker_id=worker.id, error=str(exc))
            return StreamingResponse(_stream(), media_type="text/event-stream")

        result = await proxy.forward(
            worker, "/completions", body, req_id,
            token_estimate=token_est,
            on_complete=_on_complete,
        )
        if prompt:
            trie.insert(model, prompt, worker.id)
        return JSONResponse(content=result)
    except WorkerCircuitOpenError as exc:
        return _error_response(503, "circuit_open", f"worker {exc.worker_id} circuit is open")
    except WorkerTimeoutError as exc:
        return _error_response(504, "timeout", f"worker {exc.worker_id} timed out")
    except WorkerUnavailableError as exc:
        return _error_response(503, "worker_unavailable", f"worker {exc.worker_id} unreachable")
    except WorkerHTTPError as exc:
        if 400 <= exc.status_code < 500:
            return Response(content=exc.body, status_code=exc.status_code, media_type="application/json")
        return _error_response(502, "upstream_error", f"worker {exc.worker_id} returned {exc.status_code}")


@router.post("/v1/embeddings")
async def embeddings(request: Request) -> Response:
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err
    req_id = _request_id(request)
    try:
        body = await request.json()
    except Exception:
        return _error_response(400, "invalid_request", "request body is not valid JSON")
    model = body.get("model", "")

    kv_router: KVAwareRouter = request.app.state.router
    proxy: HttpProxy = request.app.state.proxy

    try:
        worker, _ = kv_router.select(model, [])
    except NoWorkersAvailable as exc:
        return _error_response(503, "no_workers", str(exc))

    try:
        result = await proxy.forward(worker, "/embeddings", body, req_id)
        return JSONResponse(content=result)
    except WorkerCircuitOpenError as exc:
        return _error_response(503, "circuit_open", f"worker {exc.worker_id} circuit is open")
    except WorkerTimeoutError as exc:
        return _error_response(504, "timeout", f"worker {exc.worker_id} timed out")
    except WorkerUnavailableError as exc:
        return _error_response(503, "worker_unavailable", f"worker {exc.worker_id} unreachable")
    except WorkerHTTPError as exc:
        if 400 <= exc.status_code < 500:
            return Response(content=exc.body, status_code=exc.status_code, media_type="application/json")
        return _error_response(502, "upstream_error", f"worker {exc.worker_id} returned {exc.status_code}")


@router.get("/v1/models")
async def list_models(request: Request) -> Response:
    auth_err = _check_auth(request)
    if auth_err:
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
    hits: int = getattr(request.app.state, "cache_hits_total", 0)
    config: GatewayConfig = request.app.state.config
    breaker_states = proxy.breaker_states()

    worker_loads = [
        {
            "id": w.id,
            "model": w.model,
            "type": w.type,
            "role": w.role,
            "tokens_in_flight": w.tokens_in_flight,
            "cache_utilization": w.cache_utilization,
            "circuit_state": breaker_states.get(w.id, "CLOSED"),
        }
        for w in workers
    ]
    payload: dict = {
        "status": "ok",
        "workers": len(workers),
        "worker_loads": worker_loads,
        "trie_entries": trie.entry_count(),
        "cache_hits_total": hits,
        "redis_enabled": bool(config.redis_url),
        "redis_transport": config.redis_transport if config.redis_url else None,
        "disaggregation_enabled": config.disaggregation_enabled,
        "prometheus_enabled": config.prometheus_enabled,
        "otel_enabled": getattr(request.app.state, "otel_enabled", False),
    }

    sessions: SessionTracker | None = getattr(request.app.state, "sessions", None)
    if sessions is not None:
        payload["active_sessions"] = sessions.session_count()
        payload["session_counters"] = {
            "prefill_complete_received_total": sessions.prefill_complete_received_total,
            "sticky_hits_total": sessions.sticky_hits_total,
            "sticky_misses_total": sessions.sticky_misses_total,
            "session_eviction_total": sessions.session_eviction_total,
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
    sessions: SessionTracker | None = getattr(request.app.state, "sessions", None)

    metrics.trie_entries.set(trie.entry_count())
    if sessions is not None:
        metrics.active_sessions.set(sessions.session_count())

    # workers_healthy counts only workers whose breaker is not OPEN, so the
    # gauge reflects real availability rather than the static registry size.
    breaker_states = proxy.breaker_states()
    role_counts: dict[tuple[str, str], int] = {}
    for w in registry.all_workers():
        metrics.worker_tokens_in_flight.labels(worker_id=w.id).set(w.tokens_in_flight)
        metrics.worker_cache_utilization.labels(worker_id=w.id).set(w.cache_utilization)
        key = (w.model, w.role)
        role_counts.setdefault(key, 0)
        if breaker_states.get(w.id, "CLOSED") != "OPEN":
            role_counts[key] += 1
    for (model, role), count in role_counts.items():
        metrics.workers_healthy.labels(model=model, role=role).set(count)

    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)
