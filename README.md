# InferMesh

OpenAI-compatible inference gateway that routes requests by KV-cache locality.
Sits between OpenAI-format clients and any OpenAI-compatible backend (Groq,
Gemini, vLLM), and picks the worker most likely to have a warm KV cache for
each request.

## Features

- Character-level `RadixTrie` with TTL eviction for longest-prefix routing.
- Power-of-two-choices fallback scored on tokens-in-flight (ETIF) and GPU
  cache utilization.
- Optional disaggregated prefill/decode routing with session affinity via
  `X-InferMesh-Session-ID` and `prefill_complete` event handoff over Redis.
- Per-worker circuit breaker (CLOSED, OPEN, HALF_OPEN), semaphore-bounded
  concurrency, typed error propagation.
- Prometheus `/metrics` with 13 metric families. OpenTelemetry spans on the
  request path. Grafana dashboard with 20 panels in `bench/grafana/`.
- Redis Streams (at-least-once with replay) or Pub/Sub for worker events.
- React dashboard at `/`.

## Quick start

```bash
uv sync
cp .env.example .env       # add GROQ_API_KEY and/or GEMINI_API_KEY
uv run uvicorn infermesh.main:create_app --factory --port 8000
```

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-3.1-8b-instant","messages":[{"role":"user","content":"hi"}]}'
```

## Endpoints

| Path | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible chat, streaming and non-streaming. |
| `POST /v1/completions` | OpenAI-compatible completions. |
| `POST /v1/embeddings` | Pass-through. |
| `GET /v1/models` | Models registered in `workers.yaml`. |
| `GET /health` | Worker loads, circuit states, trie and session counters. |
| `GET /metrics` | Prometheus scrape. |
| `GET /` | React dashboard. |

## Architecture

| Component | Module | Notes |
|---|---|---|
| Router | `infermesh/routing.py` | `RadixTrie` (longest-prefix match, per-model roots, TTL) + `PowerOfTwoChoicesRouter` (ETIF + cache util). |
| Disaggregated router | `infermesh/routing.py` | `DisaggregatedRouter` splits prefill / decode / mixed pools. `SessionTracker` pins conversations to the decode worker holding their KV. |
| Event subscriber | `infermesh/kv_subscriber.py` | Consumes `kv_cached`, `kv_evicted`, `prefill_complete` from Redis Streams or Pub/Sub. Reconnect with exponential backoff. |
| HTTP proxy | `infermesh/proxy.py` | httpx with three-state circuit breaker, per-worker semaphore, OTEL spans, typed exceptions. |
| Metrics | `infermesh/metrics.py` | Dedicated `CollectorRegistry`; counters, histograms, gauges. |
| Tracing | `infermesh/tracing.py` | `route_selection`, `worker_request`, `kv_event.*` spans. FastAPI + httpx auto-instrumented. No-op when `OTEL_EXPORTER_OTLP_ENDPOINT` is unset. |

## Measured results

Real-API benchmarks against Groq and Gemini. Reproducible from
`bench/results/`. See [`bench/README.md`](bench/README.md) for the full
methodology and charts.

Trie warmup, 200-request Groq shared-prefix run:

```
request   1: 0%    (cold trie)
request   2: 50%
request   5: 80%
request  20: 95%
request  30: 97%   (asymptote)
```

Workload isolation, 2502 requests through Locust at 15-21 RPS sustained:

| Workload | Cache hit rate |
|---|---|
| Shared system prompt, varying user question | 99% |
| Multi-turn conversation with session header | 68% |
| Unique uuid per request | 36% |

A "cache hit" here means the router found a worker via trie prefix match (any
depth > 0 by default); it reflects routing locality, not a guaranteed warm KV
cache upstream. On workloads that merely share a system prompt this can
over-count — raise `INFERMESH_KV_MATCH_MIN_CHARS` to require a longer shared
prefix before a match counts.

Circuit breaker under contention, 200 concurrent Groq shared-prefix requests:

|  | Median | p99 |
|---|---|---|
| Through gateway | 253 ms | 607 ms |
| Direct to provider | 4564 ms | 7569 ms |

This gap is **load-shedding, not a cache speedup**. Under free-tier rate limiting
the direct client retries `429`s with exponential backoff (its latency includes
that sleep), while the gateway's circuit breaker turns overload into instant
`503`s. The headline 18× is the **median** ratio; the p99 ratio is 12.5×. With
warm backends and no contention the gateway adds only single-digit-ms proxy
overhead — it does not make an individual successful call faster.

## Configuration

All settings are environment variables prefixed `INFERMESH_`. Source:
[`infermesh/settings.py`](infermesh/settings.py).

| Variable | Default | Purpose |
|---|---|---|
| `INFERMESH_WORKERS_CONFIG_PATH` | `workers.yaml` | Worker registry path. |
| `INFERMESH_DISAGGREGATION_ENABLED` | `false` | Enable prefill/decode routing. |
| `INFERMESH_PREFIX_TTL_S` | `300` | RadixTrie entry TTL. |
| `INFERMESH_REDIS_URL` | `""` | Empty disables Redis. |
| `INFERMESH_REDIS_TRANSPORT` | `streams` | `streams` or `pubsub`. |
| `INFERMESH_ETIF_WEIGHT` | `0.6` | ETIF vs cache-util weight in P2C scoring. |
| `INFERMESH_DECODE_SESSION_TTL_S` | `600` | Sticky session TTL. |
| `INFERMESH_MAX_CONCURRENCY_PER_WORKER` | `20` | Per-worker concurrency cap. |
| `INFERMESH_METRICS_POLL_INTERVAL_S` | `10` | Upstream `/metrics` poll cadence. |
| `INFERMESH_KV_MATCH_MIN_CHARS` | `0` | Min trie prefix-match length to count as a cache hit (`0` = any match > 0). |
| `INFERMESH_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `5` | Consecutive failures before a worker circuit opens. |
| `INFERMESH_CIRCUIT_BREAKER_RECOVERY_TIMEOUT_S` | `30` | Seconds a circuit stays OPEN before a single HALF_OPEN probe. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | Enables OpenTelemetry export. |

## Observability stack

```bash
cd bench/grafana
docker compose up -d
# Grafana: http://localhost:3000 (anonymous Viewer)
```

The dashboard provisions automatically. Panels cover request rate by status,
latency percentiles via `histogram_quantile`, cache hit rate over time,
circuit-breaker opens, per-worker ETIF, healthy workers by role, Redis
event throughput.

## Tests

```bash
uv run --group dev pytest
```

Unit tests cover the RadixTrie (incl. a longest-prefix fuzz property), P2C
scoring, the circuit breaker (open / single-probe HALF_OPEN / 4xx-doesn't-trip /
ETIF-no-leak regression), the Redis event subscriber, and the OpenAI API
surface. `bench/redis_replay_demo.py` reproduces at-least-once Streams replay
against a local Redis.

## Layout

```
infermesh/     Gateway code.
frontend/     React + Vite dashboard. Built output lives in infermesh/static/.
bench/        Benchmarks (asyncio + Locust) and Grafana stack.
workers.yaml  Worker registry (Groq + Gemini by default).
pyproject.toml uv project, optional `bench` dependency group.
```
