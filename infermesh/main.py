"""FastAPI application factory."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
import structlog.stdlib
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from infermesh.proxy import HttpProxy
from infermesh.registry import WorkerRegistry, parse_gpu_cache_utilization
from infermesh.routing import (
    DisaggregatedRouter,
    KVAwareRouter,
    RadixTrie,
    SessionTracker,
)
from infermesh.routes import router
from infermesh.settings import GatewayConfig
from infermesh.tracing import init_tracing, instrument_app

_STATIC_DIR = Path(__file__).parent / "static"

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

log = structlog.get_logger()


async def _evict_trie_loop(trie: RadixTrie, interval_s: float = 60.0) -> None:
    while True:
        await asyncio.sleep(interval_s)
        removed = trie.evict_expired()
        if removed:
            log.debug("trie_eviction", removed=removed)


async def _evict_sessions_loop(
    sessions: SessionTracker, interval_s: float = 30.0
) -> None:
    while True:
        await asyncio.sleep(interval_s)
        removed = sessions.evict_expired()
        if removed:
            log.debug(
                "session_eviction", removed=removed, active=sessions.session_count()
            )


async def _poll_metrics_loop(
    registry: WorkerRegistry,
    proxy: HttpProxy,
    interval_s: float,
) -> None:
    while True:
        await asyncio.sleep(interval_s)
        for worker in registry.all_workers():
            if worker.type != "vllm":
                continue
            try:
                text = await proxy.get_raw(worker, "/metrics")
                utilization = parse_gpu_cache_utilization(text)
                registry.update_utilization(worker.id, utilization)
                log.debug("metrics_polled", worker_id=worker.id, cache_utilization=utilization)
            except Exception as exc:
                log.warning("metrics_poll_failed", worker_id=worker.id, error=str(exc))


def create_app() -> FastAPI:
    # WorkerRegistry reads worker API keys via os.environ, so we need
    # dotenv to populate the process env (pydantic-settings only loads
    # its own model fields).
    load_dotenv()
    config = GatewayConfig()

    otel_enabled = init_tracing(config.otel_service_name)
    if otel_enabled:
        log.info("otel_tracing_enabled", service=config.otel_service_name)

    registry = WorkerRegistry(config.workers_config_path)
    trie = RadixTrie(ttl_s=config.prefix_ttl_s)
    # Always instantiate KVAwareRouter for /v1/completions, /v1/embeddings,
    # and the non-disagg chat path.
    kv_router = KVAwareRouter(registry, trie, config)
    proxy = HttpProxy(config)

    sessions: SessionTracker | None = None
    disagg_router: DisaggregatedRouter | None = None
    if config.disaggregation_enabled:
        sessions = SessionTracker(ttl_s=config.decode_session_ttl_s)
        disagg_router = DisaggregatedRouter(registry, trie, sessions, config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info(
            "gateway_started",
            workers=len(registry.all_workers()),
            models=registry.all_models(),
            redis_enabled=bool(config.redis_url),
            redis_transport=config.redis_transport if config.redis_url else None,
            disaggregation_enabled=config.disaggregation_enabled,
            prometheus_enabled=config.prometheus_enabled,
        )
        evict_task = asyncio.create_task(_evict_trie_loop(trie))
        poll_task = asyncio.create_task(
            _poll_metrics_loop(registry, proxy, config.metrics_poll_interval_s)
        )
        sess_task: asyncio.Task | None = None
        if sessions is not None:
            sess_task = asyncio.create_task(_evict_sessions_loop(sessions))

        redis_task: asyncio.Task | None = None
        if config.redis_url:
            from infermesh.kv_subscriber import kv_subscriber_loop
            redis_task = asyncio.create_task(
                kv_subscriber_loop(trie, config, sessions=sessions)
            )

        try:
            yield
        finally:
            evict_task.cancel()
            poll_task.cancel()
            if sess_task is not None:
                sess_task.cancel()
            if redis_task is not None:
                redis_task.cancel()
                try:
                    await redis_task
                except asyncio.CancelledError:
                    pass
            await proxy.client.aclose()
            log.info("gateway_stopped")

    app = FastAPI(
        title="InferMesh",
        description="KV-cache-aware distributed LLM inference gateway",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.registry = registry
    app.state.trie = trie
    app.state.router = kv_router
    app.state.proxy = proxy
    app.state.cache_hits_total = 0
    app.state.sessions = sessions
    app.state.disagg_router = disagg_router
    app.state.otel_enabled = otel_enabled

    app.include_router(router)

    if otel_enabled:
        instrument_app(app)

    if _STATIC_DIR.is_dir():
        app.mount("/assets", StaticFiles(directory=_STATIC_DIR / "assets"), name="assets")

        @app.get("/", include_in_schema=False)
        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str = "") -> FileResponse:
            return FileResponse(_STATIC_DIR / "index.html")

    return app
