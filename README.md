# InferMesh

An OpenAI-compatible gateway that sends each request to the LLM worker most likely to hold its prompt prefix in cache, and keeps working when workers fail or rate-limit.

It sits between OpenAI-format clients and any OpenAI-compatible backend (Groq, Gemini, vLLM). Routing follows SGLang's cache-aware router: a radix trie of prompt prefixes, a match threshold, and a load-imbalance guard, with power-of-two choices as the fallback.

## Quick start

```bash
uv sync --group dev
cp .env.example .env        # add GROQ_API_KEY and/or GEMINI_API_KEY
uv run uvicorn infermesh.main:create_app --factory --port 8000

curl -X POST localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"openai/gpt-oss-20b","messages":[{"role":"user","content":"hi"}]}'
```

`/health` shows live worker state, `/metrics` is the Prometheus scrape. Workers are listed in `workers.yaml`.

## What it does

- **Cache-aware routing** over a per-worker capped prefix trie that learns from the requests it routes. An optional Redis consumer can also ingest worker-published prefix events; no stock engine emits that schema, so it needs a small sidecar (see [docs/configuration.md](docs/configuration.md)).
- **Failure handling:** per-worker circuit breaker, bounded wait queue, retries before the first byte, a retry budget (10% of requests by default), and `Retry-After` / reset-header cool-downs.
- **Honest load signals:** engine metrics that are missing or stale are ignored, not read as idle.
- **Observability:** 15+ Prometheus metric families, OpenTelemetry spans and a Grafana dashboard.

## Measured

On a replay of a real prefix-reuse trace (Mooncake, 8 workers, 20 seeds) the cache-aware router delivers 1.77 req/s of goodput [1.76, 1.78] against 1.39 for power-of-two choices and 1.07 for round-robin. With a fixed two-attempt retry, a mock upstream collapses near capacity; a retry budget does not. The simulator's service times are assumptions, so read the ordering, not the latencies.

An earlier "18x faster" Groq number was load shedding and has been removed. Tables and how to reproduce: [docs/benchmarks.md](docs/benchmarks.md).

## More

- [docs/configuration.md](docs/configuration.md): every `INFERMESH_*` setting, and running several replicas.
- [docs/benchmarks.md](docs/benchmarks.md): results, method, reproduction.
- `uv run python -m pytest tests/ -q` runs the tests. `bench/grafana/` has the dashboard (`docker compose up -d`).
- Layout: `infermesh/` gateway, `tests/`, `bench/` (replay and retry experiments, Grafana).
