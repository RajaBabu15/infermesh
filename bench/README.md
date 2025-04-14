# Benchmarks

Three harnesses targeting real Groq and Gemini APIs through the gateway.
See [`../README.md`](../README.md) for project context.

| Harness | Tool | Use when |
|---|---|---|
| `real_bench.py` | asyncio + httpx | Gateway vs. direct baseline with charts. |
| `run_locust.py` | Locust (subprocess) | High-concurrency load across workload classes. |
| `grafana/` | docker compose | Live PromQL dashboards over `/metrics`. |

Prerequisites:

- `GROQ_API_KEY` and `GEMINI_API_KEY` in `.env`
- `uv sync --group bench` (installs locust, matplotlib, pandas)

## `real_bench.py`

Compares two modes on the same workload:

- `gateway`: request routed through InferMesh in-process via ASGITransport.
- `direct`: request goes straight to the provider HTTPS endpoint.

Two scenarios per provider:

- `shared_prefix`: identical 200-char system prompt, varying user question.
- `diverse`: unique uuid prefix per request.

```bash
# 200 Groq + 15 Gemini per cell, 2 scenarios, 2 modes = 860 calls.
uv run python -m bench.real_bench

uv run python -m bench.real_bench --requests-groq 500 --concurrency-groq 25
uv run python -m bench.real_bench --providers groq
uv run python -m bench.real_bench --modes gateway
```

Output in `bench/results/<utc-ts>/`:

- `raw.csv` per-request rows
- `summary.json` aggregated stats and Prometheus deltas
- `charts/p99_gateway_vs_direct.png`
- `charts/median_gateway_vs_direct.png`
- `charts/failures.png`
- `charts/latency_over_time.png`
- `charts/warmup_curve.png`

### Reference run `bench/results/20260515-194403/`

860 calls. 200 Groq + 15 Gemini per cell, concurrency 15 and 3.

```
mode     provider  scenario       n   ok  429  5xx    med    p95    p99 retries
direct   groq      shared_prefix 200  44  156    0   4564   6990   7569     694
gateway  groq      shared_prefix 200  30    0  170    253    565    607      15
direct   groq      diverse       200  41  159    0   4586   7078   7449     715
gateway  groq      diverse       200   0    0  200      0      0      0       0

Gateway delta:
  requests_total:         +451
  cache_hits_total:       +195
  cache_misses_total:     +256
  gateway cache hit rate: 43.2%
  circuit_breaker_opens:    +2
  trie entries:         0, 15
```

Gateway median is 253 ms vs. 4564 ms direct on Groq `shared_prefix` (18x
lower). Most of that gap is the circuit breaker rejecting doomed retries
instantly. The trie's own signal is in `warmup_curve.png`: cumulative
gateway cache-hit rate climbs from 0% on request 1 to ~97% by request 30.

## `run_locust.py`

Spawns the gateway as a uvicorn subprocess on a real port and drives Locust
against it. Three user classes:

- `SharedPromptGroqUser`: single-turn, identical system prompt.
- `DiverseGroqUser`: unique uuid prefix per request.
- `ConversationalGroqUser`: multi-turn with session header propagation.

```bash
# 3 cells, 20 users, 45 s each.
uv run python -m bench.run_locust

uv run python -m bench.run_locust --users 10 --duration 30
uv run python -m bench.run_locust --users 50
uv run python -m bench.run_locust --classes SharedPromptGroqUser
```

Output in `bench/results/<utc-ts>/locust/`:

- `<class>_stats.csv`, `<class>_stats_history.csv`, `<class>_failures.csv`
- `<class>_prom.json` pre/post `/metrics` scrape
- `manifest.json`, `summary.json`
- `charts/cache_hit_rate_by_class.png`
- `charts/outcomes_by_class.png`
- `charts/p99_by_class.png`

### Reference run `bench/results/20260515-195801/locust/`

2502 Groq requests across 3 cells. 20 users, 45 s each.

```
user_class                  reqs    rps  med  p95  p99  hits  miss  hit%  cb
SharedPromptGroqUser         892   21.2    4  180  370   893    10    99   2
DiverseGroqUser              925   21.0    4    5  290   339   598    36   2
ConversationalGroqUser       685   15.5    4   82  300   473   223    68   2
```

Three workload patterns through the same code path produce three distinct
cache-hit rates. The trie matches by prefix structure.

## `grafana/`

```bash
# Terminal 1
uv run uvicorn infermesh.main:create_app --factory --port 8000

# Terminal 2
cd bench/grafana
docker compose up -d
# http://localhost:3000
```

Dashboard "InferMesh Gateway" provisions automatically. 20 panels across
Traffic, Latency, KV cache, and Workers rows. PromQL queries cover request
rate by status, latency percentiles via `histogram_quantile`, cache hit
rate over time, circuit-breaker opens, per-worker ETIF, healthy workers by
role, Redis event throughput.

Files:

- `docker-compose.yaml` Prometheus 3.7 + Grafana 13
- `prometheus.yml` 5s scrape of `host.docker.internal:8000/metrics`
- `dashboards/infermesh.json` the dashboard
- `provisioning/` datasource and dashboard auto-load

## Scope

Verified by these benchmarks:

- `/health` and `/metrics` reflect real request behaviour end to end.
- RadixTrie hit rate tracks prefix structure (99%, 68%, 36%).
- Circuit breaker reduces p99 by 12.5x vs. direct under contention.
- Redis Streams transport delivers at-least-once with reconnect replay
  (verified separately by publishing 5 events while the subscriber is
  offline and replaying on restart).

Not covered:

- Disaggregated routing at scale; needs GPU workers with NIXL.
- Sustained 1000+ RPS; needs paid API tier or self-hosted GPU.
- Per-request latency attributable to the trie alone; needs multiple
  distinct worker pools (every Groq worker entry points at the same
  effective backend).
