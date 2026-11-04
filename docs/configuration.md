# Configuration and replicas

## Running more than one gateway replica

The prefix trie lives in each replica's memory. Give every replica its own
`INFERMESH_REDIS_STREAM_CONSUMER_NAME` and its own consumer group
(`INFERMESH_REDIS_STREAM_CONSUMER_GROUP`) so each one receives every event, and
let a restarted replica rebuild by replaying the stream's pending entries. Events
may carry a per-worker `seq`: a gap drops that worker's entries instead of
trusting an incomplete index. Sharing the trie between replicas is not implemented.

## Configuration

All settings use the `INFERMESH_` env prefix. Source:
[`infermesh/settings.py`](../infermesh/settings.py).

| Variable | Default | Purpose |
|---|---|---|
| `INFERMESH_WORKERS_CONFIG_PATH` | `workers.yaml` | Worker registry path. |
| `INFERMESH_PREFIX_TTL_S` | `300` | RadixTrie entry TTL. |
| `INFERMESH_REDIS_URL` | `""` | Empty disables Redis subscriber. |
| `INFERMESH_REDIS_TRANSPORT` | `streams` | `streams` (at-least-once + PEL replay) or `pubsub`. |
| `INFERMESH_ETIF_WEIGHT` | `0.6` | ETIF vs cache-util weight in P2C scoring. |
| `INFERMESH_ETIF_SCALE` | `0` (auto) | Tokens in flight at which a worker counts as fully loaded. 0 = `max_concurrency_per_worker * 1024` (about 1k tokens per request). Was a fixed 4096. |
| `INFERMESH_QUEUE_WEIGHT` / `_QUEUE_SCALE` | `0` / `8` | Weight of `vllm:num_requests_waiting` in the P2C score, saturating at the scale. 0 keeps the old score. |
| `INFERMESH_ETIF_COUNT_OUTPUT_TOKENS` | `true` | Reserve the client's `max_tokens` in ETIF until the response ends. |
| `INFERMESH_METRICS_MAX_AGE_S` | `0` (auto) | Engine metrics older than this are ignored. 0 = three poll intervals. |
| `INFERMESH_PREFIX_MAX_CHARS_PER_WORKER` | `67108864` | Trie size cap per worker; least-recently-used leaves are evicted. |
| `INFERMESH_RETRY_BUDGET_MODE` | `static` | `off`, `static` (10% of requests) or `headroom` (experimental, needs `RETRY_QUOTA_RPS`). |
| `INFERMESH_RETRY_BUDGET_PERCENT` / `_BURST_CREDITS` / `_MAX_CREDITS` | `0.10` / `3` / `10` | Credits deposited per request, initial credits, ledger cap. |
| `INFERMESH_RETRY_RHO_BASE` / `_ALPHA` | `1` / `0` | Retry cost `rho_base * (1 - alpha * prefix_hit_ratio)`. Keep `alpha = 0` until measured. |
| `INFERMESH_MAX_QUEUE_PER_WORKER` / `_QUEUE_TIMEOUT_S` | `40` / `30` | Requests allowed to wait for a worker slot, and how long. |
| `INFERMESH_RETRY_AFTER_CAP_S` | `300` | Longest cool-down honoured from `Retry-After` or reset headers. |
| `INFERMESH_CIRCUIT_BREAKER_MAX_RECOVERY_TIMEOUT_S` | `300` | Cap for Envoy ejection time (`base * consecutive opens`). |
| `INFERMESH_CACHE_THRESHOLD` | `0.3` | Trie match must cover more than this fraction of the prompt (SGLang default). |
| `INFERMESH_KV_MATCH_MIN_CHARS` | `0` | Extra absolute floor on match length, in characters. |
| `INFERMESH_BALANCE_ABS_THRESHOLD` | `64` | Skip the cache if max − min requests in flight exceeds this… |
| `INFERMESH_BALANCE_REL_THRESHOLD` | `1.5` | …and max > min × this (SGLang defaults). |
| `INFERMESH_MAX_ATTEMPTS` | `2` | Attempts across distinct workers of a model (429, 5xx, connection errors, open circuit). |
| `INFERMESH_MAX_CONCURRENCY_PER_WORKER` | `20` | Per-worker semaphore cap. |
| `INFERMESH_METRICS_POLL_INTERVAL_S` | `10` | vLLM upstream `/metrics` poll cadence. |
| `INFERMESH_CIRCUIT_BREAKER_FAILURE_THRESHOLD` | `5` | Consecutive 5xx/transport failures before OPEN. |
| `INFERMESH_CIRCUIT_BREAKER_RECOVERY_TIMEOUT_S` | `30` | Seconds OPEN before one HALF_OPEN probe (a 429 uses its `Retry-After` instead). |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | Enables OpenTelemetry export. |

Default `workers.yaml` registers **4 workers** (2 Groq, 2 Gemini), all
`type: openai`.

## Worker events (optional)

With `INFERMESH_REDIS_URL` set, the gateway consumes a Redis stream (or pub/sub channel) of JSON events that tell it what each worker holds: `kv_cached` and `kv_evicted` (`worker_id`, `model`, `prefix` text), and `kv_cleared` (`worker_id`) when an engine flushes its cache. An optional per-worker `seq` lets the gateway detect lost or replayed events. No inference engine publishes this schema natively (vLLM's KV events carry block hashes, not prefix text), so a sidecar has to translate. Nothing in the benchmarks depends on it; the replay uses the gateway's own request history.
