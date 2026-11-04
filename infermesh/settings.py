from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkerConfig(BaseModel):
    id: str
    type: Literal["openai", "vllm"] = "openai"
    url: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    weight: int = Field(default=1, ge=1)
    api_key: str = ""
    # None = no engine metric yet (never polled). Unknown is not idle: the router
    # then scores the worker on in-flight load alone.
    cache_utilization: float | None = Field(default=None, ge=0.0, le=1.0)
    cache_utilization_at: float | None = None      # monotonic time of the last good poll
    queue_depth: int | None = Field(default=None, ge=0)   # vllm:num_requests_waiting
    tokens_in_flight: int = 0
    requests_in_flight: int = 0

    def reserve(self, tokens: int) -> None:
        self.tokens_in_flight += tokens
        self.requests_in_flight += 1

    def release(self, tokens: int) -> None:
        self.tokens_in_flight = max(0, self.tokens_in_flight - tokens)
        self.requests_in_flight = max(0, self.requests_in_flight - 1)


class GatewayConfig(BaseSettings):
    api_key: str = ""
    workers_config_path: str = "workers.yaml"
    log_level: str = "INFO"
    httpx_timeout_s: float = 120.0
    max_concurrency_per_worker: int = Field(default=20, ge=1)
    prefix_ttl_s: float = Field(default=300.0, ge=10.0)
    # Record a request's prefix in the trie when it is routed, not only when it
    # finishes (SGLang's router inserts at selection time). Without it, requests that
    # share a prefix and arrive while the first is still prefilling scatter across
    # workers and each pays a cold prefill.
    insert_on_dispatch: bool = True
    # Characters of prefix the trie may attribute to one worker. SGLang's router
    # caps its tree at 2**26 (max_tree_size); least-recently-used leaves go first.
    prefix_max_chars_per_worker: int = Field(default=2**26, ge=0)
    metrics_poll_interval_s: float = Field(default=10.0, ge=1.0)
    # Engine metrics older than this are ignored (the worker is scored on ETIF
    # alone) instead of being trusted forever. 0 = three poll intervals.
    metrics_max_age_s: float = Field(default=0.0, ge=0.0)
    etif_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    # Tokens in flight at which a worker counts as fully loaded. 0 = auto:
    # max_concurrency_per_worker * 1024 (assumes about 1k tokens per request, the
    # size _estimate_tokens gives a short chat). The old fixed 4096 saturated at
    # four such requests, far below the 20-request concurrency cap.
    etif_scale: int = Field(default=0, ge=0)
    # Weight of the engine queue depth (vllm:num_requests_waiting) in the load
    # score, saturating at queue_scale waiting requests. 0 keeps the old score.
    queue_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    queue_scale: int = Field(default=8, ge=1)
    # ETIF reserves prompt tokens plus the client's max_tokens until the response
    # ends. Set false to reserve prompt tokens only.
    etif_count_output_tokens: bool = True
    etif_default_output_tokens: int = Field(default=256, ge=0)
    kv_match_min_chars: int = Field(default=0, ge=0)
    # Cache-aware routing thresholds. Defaults are SGLang router's CLI defaults
    # (--cache-threshold 0.3, --balance-abs-threshold 64,
    # --balance-rel-threshold 1.5); load is requests in flight.
    cache_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    balance_abs_threshold: int = Field(default=64, ge=0)
    balance_rel_threshold: float = Field(default=1.5, ge=1.0)
    # Attempts per request across distinct workers of the same model. A retry
    # happens only before any byte reached the client.
    max_attempts: int = Field(default=2, ge=1)
    # Retry admission (infermesh/retry_budget.py). A retry costs
    # rho = rho_base * (1 - alpha * prefix_hit_ratio_of_fallback) credits;
    # each request deposits retry_budget_percent credits (SRE book: 10 percent).
    retry_budget_mode: Literal["off", "static", "headroom"] = "static"
    retry_budget_percent: float = Field(default=0.10, ge=0.0, le=1.0)
    retry_burst_credits: float = Field(default=3.0, ge=0.0)
    retry_max_credits: float = Field(default=10.0, gt=0.0)
    retry_rho_base: float = Field(default=1.0, gt=0.0)
    retry_alpha: float = Field(default=0.0, ge=0.0, le=1.0)
    # Shared upstream quota in requests per second; used only in headroom mode.
    retry_quota_rps: float = Field(default=0.0, ge=0.0)
    retry_rate_window_s: float = Field(default=10.0, gt=0.0)
    circuit_breaker_failure_threshold: int = Field(default=5, ge=1)
    circuit_breaker_recovery_timeout_s: float = Field(default=30.0, ge=1.0)
    # Each consecutive ejection doubles the recovery time, up to this cap
    # (Envoy's max_ejection_time default is also 300 s).
    circuit_breaker_max_recovery_timeout_s: float = Field(default=300.0, ge=1.0)
    # Longest cool-down honoured from Retry-After / x-ratelimit-reset-* hints.
    retry_after_cap_s: float = Field(default=300.0, ge=1.0)
    # Requests waiting for a worker slot beyond the semaphore. When full the
    # gateway answers 503 with Retry-After instead of queueing without limit.
    max_queue_per_worker: int = Field(default=40, ge=0)
    queue_timeout_s: float = Field(default=30.0, gt=0.0)
    redis_url: str = ""
    redis_channel: str = "infermesh:kv_events"
    redis_event_ttl_s: float = Field(default=3600.0, ge=10.0)


    # "streams" gives at-least-once with reconnect replay. "pubsub" does not.
    redis_transport: Literal["pubsub", "streams"] = "streams"
    redis_stream_consumer_group: str = "infermesh-gateway"
    redis_stream_consumer_name: str = "gateway-0"

    # OTEL tracing activates only when OTEL_EXPORTER_OTLP_ENDPOINT is set.
    otel_service_name: str = "infermesh-gateway"

    @property
    def effective_etif_scale(self) -> int:
        return self.etif_scale or self.max_concurrency_per_worker * 1024

    @property
    def effective_metrics_max_age_s(self) -> float:
        return self.metrics_max_age_s or 3 * self.metrics_poll_interval_s

    model_config = SettingsConfigDict(env_prefix="INFERMESH_", env_file=".env", extra="ignore")
