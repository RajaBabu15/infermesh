# InferMesh

OpenAI-compatible inference gateway that routes requests by KV-cache locality.
Sits between OpenAI-format clients and any OpenAI-compatible backend (Groq,
Gemini, vLLM), and picks the worker most likely to have a warm KV cache for
each request.

## Features

- Character-level `RadixTrie` with TTL eviction for longest-prefix routing,
  following SGLang's cache-aware router: every node records which workers sent
  that prefix, a match counts only if it covers more than `cache_threshold`
  (30%) of the prompt, and a load-imbalance guard skips the cache when one
  worker is far busier than another.
- Power-of-two-choices fallback scored on tokens-in-flight (ETIF) and vLLM KV
  cache utilization.
- Optional disaggregated prefill/decode routing with session affinity via
  `X-InferMesh-Session-ID` and `prefill_complete` event handoff over Redis.
- Per-worker circuit breaker (CLOSED, OPEN, HALF_OPEN with single-probe
  recovery), semaphore-bounded concurrency, typed error propagation. A 429
  cools the worker down for its `Retry-After`. Failed requests are retried on
  another worker of the same model before any byte reaches the client.
- Prometheus `/metrics` with **15 metric families**, including provider-reported
  prompt-cache hits (`usage.prompt_tokens_details.cached_tokens`). OpenTelemetry spans on the
  request path. Grafana dashboard with **20 panels** in `bench/grafana/`.
- Redis Streams (at-least-once with PEL replay on reconnect) or Pub/Sub for
  worker events.
- React admin UI at `/` with live ETIF, cache utilization, and circuit-breaker
  state (5 s poll via `/health`).

## Quick start

```bash
# Python gateway
uv sync --group dev
cp .env.example .env       # add GROQ_API_KEY and/or GEMINI_API_KEY

# React dashboard (optional — pre-built assets ship in infermesh/static/)
cd frontend && npm ci && npm run build && cd ..

uv run uvicorn infermesh.main:create_app --factory --host 127.0.0.1 --port 8000
```

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"openai/gpt-oss-20b","messages":[{"role":"user","content":"hi"}]}'
```

Open http://127.0.0.1:8000/ for the dashboard, http://127.0.0.1:8000/health for
live worker state, http://127.0.0.1:8000/metrics for Prometheus.

### Clean build from scratch

```bash
rm -rf .venv frontend/node_modules infermesh/static/assets infermesh/static/index.html
uv venv && uv sync --group dev --group bench
cd frontend && npm ci && npm run build && cd ..
uv run pytest tests/ -q          # expect 41 passed
```

If port 8000 is busy: `lsof -ti :8000 | xargs kill -9`

## Endpoints

| Path | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible chat, streaming and non-streaming. |
| `POST /v1/completions` | OpenAI-compatible completions. |
| `POST /v1/embeddings` | Pass-through. |
| `GET /v1/models` | Models registered in `workers.yaml`. |
| `GET /health` | Worker ETIF, circuit states, trie entries, session/Redis counters. |
| `GET /metrics` | Prometheus scrape (15 metric families). |
| `GET /` | React dashboard. |

## Architecture

| Component | Module | Notes |
|---|---|---|
| Router | `infermesh/routing.py` | `RadixTrie` (longest-prefix match, per-model roots, per-worker TTL on every node) + SGLang-style cache-aware pick (match ratio, load-imbalance guard, healthy workers only) + `PowerOfTwoChoicesRouter` (ETIF + cache util). |
| Disaggregated router | `infermesh/routing.py` | `DisaggregatedRouter` splits prefill / decode / mixed pools. `SessionTracker` pins conversations to the decode worker holding their KV. |
| Event subscriber | `infermesh/kv_subscriber.py` | Consumes `kv_cached`, `kv_evicted`, `prefill_complete` from Redis Streams or Pub/Sub. On reconnect, drains the pending-entries list (`id "0"`) before reading new entries (`">"`). |
| HTTP proxy | `infermesh/proxy.py` | httpx with three-state circuit breaker, per-worker semaphore, OTEL spans, typed exceptions. **5xx and transport errors** count toward the breaker; a **429** opens it for the upstream's `Retry-After`; other **4xx (e.g. 401)** do not. Streams are opened and status-checked before the gateway answers, so upstream errors keep their status. |
| Metrics | `infermesh/metrics.py` | Dedicated `CollectorRegistry`; 8 counters, 2 histograms, 5 gauges. `workers_healthy` excludes workers whose circuit is OPEN. |
| Tracing | `infermesh/tracing.py` | `route_selection`, `worker_request`, `kv_event.*` spans. FastAPI + httpx auto-instrumented. No-op when `OTEL_EXPORTER_OTLP_ENDPOINT` is unset. |

## Measured results

Real-API benchmarks against Groq and Gemini. Reproducible artifacts live in
`bench/results/`. Full harness docs: [`bench/README.md`](bench/README.md).

> The GitHub repository was created on 2026-05-19. The first pushed commit, `0d8ed4b`, is an "Initial Commit" authored 2026-05-15, the night of the Locust run in `bench/results/20260515-195801`, and committed on 2026-06-23. That root was replaced the same hour by history whose author dates say spring 2025 and whose seconds are `:00`. Those trees contain a Locust epoch of 2026-05-15 and npm packages published in May 2026, including `rollup@4.60.4` and `@tanstack/react-query@5.100.10`. The 2025 dates are not when those commits were written. The benchmark JSON in the tree is the record of the run. I am not dating this project 2025.

> These runs predate the routing and failover fixes, and used Groq's Llama
> models, which Groq shut down on 2026-08-16. Re-run before quoting them. The
> trie hit rates below are routing predictions; the gateway now also exports
> `infermesh_upstream_cached_tokens_total`, the provider's own count of
> cached prompt tokens.

Reference run **`bench/results/20260515-194403/`** — 200 Groq requests per cell,
concurrency 15, shared-prefix scenario:

| Mode | Median | p99 | Success rate |
|---|---|---|---|
| Through gateway | 253 ms | 607 ms | 30 / 200 |
| Direct to provider | 4564 ms | 7569 ms | 44 / 200 |
| **Median ratio** | **18.0×** | | |
| **p99 ratio** | | **12.5×** | |

This gap is **load-shedding under free-tier rate limits, not a cache speedup**.
The direct client retries `429`s with exponential backoff; the gateway's circuit
breaker turns overload into fast `503`s. With warm backends and no contention
the gateway adds only single-digit-ms proxy overhead.

Reference run **`bench/results/20260515-195801/locust/`** — 2502 Groq requests,
20 users × 45 s, three workload classes:

| Workload | Cache hit rate | p99 | RPS |
|---|---|---|---|
| Shared system prompt, varying user question | **98.9%** | 370 ms | 21.2 |
| Multi-turn conversation (session header) | **68.0%** | 300 ms | 15.5 |
| Unique uuid per request | **36.2%** | 290 ms | 21.0 |

Trie warmup curve (gateway shared-prefix, successful requests only):

```
request   1: 0%    (cold trie)
request   2: 50%
request   5: 80%
request  20: 95%
request  30: 97%   (asymptote)
```

Routing-only simulation (no upstream API, trie insert after each route):

| Pattern | n | Hit rate |
|---|---|---|
| Shared system prompt + 15-question pool | 892 | **98.9%** |
| Shared system prompt + 15-question pool | 200 | **95.5%** |
| Unique uuid-prefixed system prompt | 200 | **33.0%** |
| Fully unique user content (shared system prompt only) | 200 | **31.5%** |

The bottom two rows differ by only ~1.5pp despite carrying no reusable content
beyond the system prompt — evidence that shallow matches inflate the rate, which
is what `INFERMESH_KV_MATCH_MIN_CHARS` exists to suppress.

A "cache hit" means the router matched a worker via trie prefix covering more
than `INFERMESH_CACHE_THRESHOLD` (default 0.3) of the prompt, and at least
`INFERMESH_KV_MATCH_MIN_CHARS`. It reflects routing locality, not a guaranteed
warm KV cache upstream; `infermesh_upstream_cached_tokens_total` is the
provider-reported figure.

### Reproduce benchmarks

```bash
uv sync --group bench
# needs valid GROQ_API_KEY / GEMINI_API_KEY in .env

uv run python -m bench.real_bench --providers groq --requests-groq 200 --concurrency-groq 15
uv run python -m bench.run_locust --users 20 --duration 45
```

## Configuration

All settings use the `INFERMESH_` env prefix. Source:
[`infermesh/settings.py`](infermesh/settings.py).

| Variable | Default | Purpose |
|---|---|---|
| `INFERMESH_WORKERS_CONFIG_PATH` | `workers.yaml` | Worker registry path. |
| `INFERMESH_DISAGGREGATION_ENABLED` | `false` | Enable prefill/decode routing. |
| `INFERMESH_PREFIX_TTL_S` | `300` | RadixTrie entry TTL. |
| `INFERMESH_REDIS_URL` | `""` | Empty disables Redis subscriber. |
| `INFERMESH_REDIS_TRANSPORT` | `streams` | `streams` (at-least-once + PEL replay) or `pubsub`. |
| `INFERMESH_ETIF_WEIGHT` | `0.6` | ETIF vs cache-util weight in P2C scoring. |
| `INFERMESH_ETIF_SCALE` | `4096` | ETIF normalization denominator in P2C. |
| `INFERMESH_CACHE_THRESHOLD` | `0.3` | Trie match must cover more than this fraction of the prompt (SGLang default). |
| `INFERMESH_KV_MATCH_MIN_CHARS` | `0` | Extra absolute floor on match length, in characters. |
| `INFERMESH_BALANCE_ABS_THRESHOLD` | `64` | Skip the cache if max − min requests in flight exceeds this… |
| `INFERMESH_BALANCE_REL_THRESHOLD` | `1.5` | …and max > min × this (SGLang defaults). |
| `INFERMESH_MAX_ATTEMPTS` | `2` | Attempts across distinct workers of a model (429, 5xx, connection errors, open circuit). |
| `INFERMESH_DECODE_SESSION_TTL_S` | `600` | Sticky session TTL. |
| `INFERMESH_MAX_CONCURRENCY_PER_WORKER` | `20` | Per-worker semaphore cap. |
| `INFERMESH_METRICS_POLL_INTERVAL_S` | `10` | vLLM upstream `/metrics` poll cadence. |
| `INFERMESH_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `5` | Consecutive 5xx/transport failures before OPEN. |
| `INFERMESH_CIRCUIT_BREAKER_RECOVERY_TIMEOUT_S` | `30` | Seconds OPEN before one HALF_OPEN probe (a 429 uses its `Retry-After` instead). |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | Enables OpenTelemetry export. |

Default `workers.yaml` registers **4 workers** (2 Groq, 2 Gemini), all
`type: openai` / `role: mixed`.

## Observability stack

```bash
# Terminal 1
uv run uvicorn infermesh.main:create_app --factory --port 8000

# Terminal 2
cd bench/grafana && docker compose up -d
# Grafana: http://localhost:3000 (anonymous Viewer)
```

Dashboard **InferMesh Gateway** provisions automatically — **20 panels** (4 row
headers + 16 visualizations) across Traffic, Latency, KV cache routing, and
Workers. PromQL covers request rate by status, `histogram_quantile` latency,
cache hit rate, circuit-breaker opens, per-worker ETIF, healthy workers by role,
Redis event throughput.

## Tests

```bash
uv sync --group dev
uv run pytest tests/ -q
```

Redis Streams replay demo (needs local Redis):

```bash
redis-server --port 6379 --save "" &
uv run python bench/redis_replay_demo.py
```

## Layout

```
infermesh/     Gateway code (Python 3.11+).
frontend/      React + Vite dashboard → built into infermesh/static/.
tests/         pytest unit + in-process integration tests (41 cases).
bench/         Benchmarks (asyncio + Locust), Grafana stack, redis_replay_demo.
workers.yaml   Worker registry (Groq + Gemini by default).
pyproject.toml uv project; optional `dev` and `bench` dependency groups.
```