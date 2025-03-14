"""Prometheus metric registry.

Uses a dedicated CollectorRegistry to avoid pollution from the
prometheus_client default global registry under multi-process workers.
"""
from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry()


# ---------------------------------------------------------------------------
# Counters (monotonically increasing)
# ---------------------------------------------------------------------------

requests_total = Counter(
    "infermesh_requests_total",
    "Total requests handled by the gateway",
    ["endpoint", "model", "status"],
    registry=REGISTRY,
)

kv_cache_hits_total = Counter(
    "infermesh_kv_cache_hits_total",
    "KV cache hits broken down by source (trie / redis_event / session_sticky)",
    ["model", "source"],
    registry=REGISTRY,
)

kv_cache_misses_total = Counter(
    "infermesh_kv_cache_misses_total",
    "KV cache misses (request routed via load balancing rather than prefix match)",
    ["model"],
    registry=REGISTRY,
)

circuit_breaker_opens_total = Counter(
    "infermesh_circuit_breaker_opens_total",
    "Circuit breaker transitions to OPEN state per worker",
    ["worker_id"],
    registry=REGISTRY,
)

redis_events_received_total = Counter(
    "infermesh_redis_events_received_total",
    "Redis events received by the subscriber, by type",
    ["event_type"],
    registry=REGISTRY,
)

redis_events_dropped_total = Counter(
    "infermesh_redis_events_dropped_total",
    "Redis events dropped due to malformed payloads or unknown types",
    ["reason"],
    registry=REGISTRY,
)


# ---------------------------------------------------------------------------
# Histograms (latency distributions)
# ---------------------------------------------------------------------------

# LLM requests range from ~10ms (cached) to >60s (long generations). The
# default Prometheus buckets are too narrow.
_REQUEST_BUCKETS = (
    0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
)

# Routing decisions are sub-millisecond.
_ROUTING_BUCKETS = (
    0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1.0,
)

request_duration_seconds = Histogram(
    "infermesh_request_duration_seconds",
    "End-to-end request duration measured at the gateway",
    ["endpoint", "model", "worker_id"],
    registry=REGISTRY,
    buckets=_REQUEST_BUCKETS,
)

routing_decision_seconds = Histogram(
    "infermesh_routing_decision_seconds",
    "Time spent in router selection logic (excludes upstream call)",
    ["router"],
    registry=REGISTRY,
    buckets=_ROUTING_BUCKETS,
)


# ---------------------------------------------------------------------------
# Gauges (point-in-time)
# ---------------------------------------------------------------------------

active_sessions = Gauge(
    "infermesh_active_sessions",
    "Active decode sessions tracked by SessionTracker",
    registry=REGISTRY,
)

trie_entries = Gauge(
    "infermesh_trie_entries",
    "Live (non-expired) entries in the RadixTrie",
    registry=REGISTRY,
)

worker_tokens_in_flight = Gauge(
    "infermesh_worker_tokens_in_flight",
    "ETIF (estimated tokens in flight) per worker",
    ["worker_id"],
    registry=REGISTRY,
)

worker_cache_utilization = Gauge(
    "infermesh_worker_cache_utilization",
    "Per-worker vllm:gpu_cache_usage_perc value (0.0-1.0)",
    ["worker_id"],
    registry=REGISTRY,
)

workers_healthy = Gauge(
    "infermesh_workers_healthy",
    "Count of healthy workers per model and role",
    ["model", "role"],
    registry=REGISTRY,
)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def render() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
