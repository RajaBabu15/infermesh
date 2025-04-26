from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerConfig(BaseModel):
    id: str
    type: Literal["openai", "vllm"] = "vllm"
    url: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    weight: int = Field(default=1, ge=1)
    api_key: str = ""
    cache_utilization: float = Field(default=0.0, ge=0.0, le=1.0)
    tokens_in_flight: int = 0
    # Ignored when disaggregation_enabled is False.
    role: Literal["prefill", "decode", "mixed"] = "mixed"


class GatewayConfig(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str = ""
    workers_config_path: str = "workers.yaml"
    log_level: str = "INFO"
    httpx_timeout_s: float = 120.0
    max_concurrency_per_worker: int = Field(default=20, ge=1)
    prefix_ttl_s: float = Field(default=300.0, ge=10.0)
    metrics_poll_interval_s: float = Field(default=10.0, ge=1.0)
    etif_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    etif_scale: int = Field(default=4096, ge=256)
    # Minimum trie prefix-match length (chars) for a request to count as a
    # cache hit. 0 keeps the historical behavior (any match depth > 0 hits);
    # raising it suppresses coincidental shallow matches so the hit metric
    # tracks genuine prefix reuse rather than e.g. a shared system prompt.
    kv_match_min_chars: int = Field(default=0, ge=0)

    # Per-worker circuit breaker tuning (previously hardcoded).
    circuit_breaker_failure_threshold: int = Field(default=5, ge=1)
    circuit_breaker_recovery_timeout_s: float = Field(default=30.0, ge=1.0)
    redis_url: str = ""
    redis_channel: str = "infermesh:kv_events"
    redis_event_ttl_s: float = Field(default=3600.0, ge=10.0)

    # When False, all workers are treated as "mixed".
    disaggregation_enabled: bool = False
    decode_session_ttl_s: float = Field(default=600.0, ge=10.0)
    session_header_name: str = "X-InferMesh-Session-ID"

    # "streams" gives at-least-once with reconnect replay. "pubsub" does not.
    redis_transport: Literal["pubsub", "streams"] = "streams"
    redis_stream_consumer_group: str = "infermesh-gateway"
    redis_stream_consumer_name: str = "gateway-0"

    prometheus_enabled: bool = True
    # OTEL tracing activates only when OTEL_EXPORTER_OTLP_ENDPOINT is set.
    otel_service_name: str = "infermesh-gateway"

    model_config = SettingsConfigDict(env_prefix="INFERMESH_", env_file=".env", extra="ignore")
