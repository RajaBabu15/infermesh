# Benchmarks

## Replay on a real prefix-reuse trace

`bench/replay/` replays the Mooncake FAST'25 `conversation_trace` (first 3,000 of
12,031 requests, 8 workers) through a deterministic discrete-event cluster. The
trace carries no text, only 512-token block hashes, so each request is rendered
as exactly `4 * input_length` characters (2048 per shared block, a unique tail
for the partial last block). The policies are the gateway's own
`KVAwareRouter`, `PowerOfTwoChoicesRouter` and `RadixTrie`. Goodput is requests
meeting TTFT <= 2 s and TPOT <= 0.1 s per second (TPOT is skipped for one-token
outputs). 20 seeds, 95% percentile-bootstrap intervals.

| policy | seeds | TTFT p50 (s) | TTFT p99 (s) | goodput (req/s) | cached-token ratio |
|---|---|---|---|---|---|
| round_robin | 1 | 5.98 [5.98, 5.98] | 81.1 [81.1, 81.1] | 1.07 [1.07, 1.07] | 0.093 [0.093, 0.093] |
| least_loaded | 1 | 2.47 [2.47, 2.47] | 52.9 [52.9, 52.9] | 1.34 [1.34, 1.34] | 0.085 [0.085, 0.085] |
| p2c | 20 | 2.30 [2.24, 2.35] | 44.6 [43.9, 45.2] | 1.39 [1.38, 1.40] | 0.091 [0.088, 0.094] |
| consistent_hash | 1 | 2449.42 [2449.42, 2449.42] | 4548.2 [4548.2, 4548.2] | 0.00 [0.00, 0.00] | 0.090 [0.090, 0.090] |
| cache_aware | 20 | 1.26 [1.24, 1.29] | 43.2 [42.7, 43.7] | 1.77 [1.76, 1.78] | 0.300 [0.300, 0.301] |

`consistent_hash` hashes the first block; almost every request in this trace
shares block 0 (a common system prompt), so it sends everything to one worker.

**Read the ordering, not the absolute numbers.** The service-time constants are
assumptions: DistServe (arXiv 2401.09670, Appendix A) publishes the form of the
latency model but fits its constants per model and GPU. The 2 s TTFT SLO is a free
parameter; DistServe's 0.25 s chatbot SLO (Table 1) is for short prompts, and the
mean prompt here is 12k tokens.

**Stale load signals** (E2). Scoring every worker by expected TTFT (engine
backlog plus the prefill of the uncached part) is strong when the backlog is
live and herds when it is a stale snapshot, as the old-information literature
predicts. Softmax sampling with temperature `tau = c * lambda * T * s_bar`
(`c` = 1, untuned) beats power-of-two choices at T = 5 s outside the intervals
and only marginally at T = 20 s, and it does **not** beat the shipped
`cache_aware` policy (goodput 1.77), which uses the gateway's own live
in-flight count. So the softmax router is not in this PR.

| staleness | policy | goodput (req/s) | TTFT p99 (s) |
|---|---|---|---|
| T=0 s | ttft_argmin | 2.37 [2.37, 2.37] | 26.0 [26.0, 26.0] |
| T=0 s | ttft_p2c | 1.60 [1.58, 1.61] | 35.5 [34.9, 36.1] |
| T=0 s | ttft_softmax | 2.37 [2.37, 2.37] | 26.0 [26.0, 26.0] |
| T=5 s | ttft_argmin | 0.70 [0.70, 0.70] | 79.4 [79.4, 79.4] |
| T=5 s | ttft_p2c | 1.43 [1.41, 1.44] | 47.4 [46.6, 48.2] |
| T=5 s | ttft_softmax | 1.67 [1.65, 1.68] | 43.2 [42.2, 44.2] |
| T=20 s | ttft_argmin | 0.29 [0.29, 0.29] | 165.4 [165.4, 165.4] |
| T=20 s | ttft_p2c | 1.22 [1.21, 1.24] | 57.3 [55.1, 60.1] |
| T=20 s | ttft_softmax | 1.26 [1.25, 1.28] | 53.4 [51.6, 55.2] |

```bash
export MOONCAKE_TRACE_DIR=...   # folder with conversation_trace.jsonl (kvcache-ai/Mooncake, Apache-2.0)
uv run python -m bench.replay.run --trace $MOONCAKE_TRACE_DIR/conversation_trace.jsonl --experiment policies --limit 3000 --workers 8 --seeds 20
uv run python -m bench.replay.run --trace $MOONCAKE_TRACE_DIR/conversation_trace.jsonl --experiment staleness --limit 3000 --workers 8 --seeds 20
```

The runs are seeded and deterministic: rerunning reproduces these tables exactly (checked). Output goes to `bench/results/`, which is gitignored.

## Retry budget experiment (E1)

`bench/retry/` reproduces the metastable failure of Bronson et al., HotOS 2021,
Sec 2.1: a pool with 300 rps capacity serving 280 rps, clients that time out after
1 s and retry, and a 10 s outage. A timed-out attempt still occupies the server, so
the retry surge keeps latency above the timeout and every attempt keeps failing.
20 seeds, 95% intervals.

**Part A, one pool, the whole pool is out for 10 s.** Fixed two attempts collapses at
high load; a budget does not. "Window attempts per request" is the amplification during
the outage.

| load (rps) | policy | steady goodput (rps) | recovered | window attempts / request |
|---|---|---|---|---|
| 280 | fixed2 | 253 [210, 281] | 0.90 [0.75, 1.00] | 2.000 [2.000, 2.000] |
| 280 | static10 | 280 [279, 281] | 1.00 [1.00, 1.00] | 1.103 [1.103, 1.103] |
| 280 | headroom | 280 [279, 281] | 1.00 [1.00, 1.00] | 1.018 [1.013, 1.025] |
| 295 | fixed2 | 133 [74, 192] | 0.45 [0.25, 0.65] | 2.000 [2.000, 2.000] |
| 295 | static10 | 295 [294, 297] | 1.00 [1.00, 1.00] | 1.103 [1.102, 1.103] |
| 295 | headroom | 295 [294, 297] | 1.00 [1.00, 1.00] | 1.002 [1.001, 1.003] |

Both budgets recover every time, so steady goodput cannot tell them apart. They differ in
the wasted work: near capacity the headroom budget sends 1.002 attempts per request at
295 rps against 1.103 for the static budget, because it deposits only
`Q / lambda - 1` credits and that is about zero when load is close to the quota. The price
is about one point of window success (0.000 against 0.010). Static stays the default;
headroom (needs `INFERMESH_RETRY_QUOTA_RPS`) is the option for a shared quota that
retries must not exhaust.

**Part B, the cache-cost link.** Four servers with prefix affinity; a warm attempt costs
20% of a cold one; **only server 0 is out**, so retries fail over to servers 1 to 3,
which then hold the prefix warm. Success ratio of requests that arrive during the outage:

| policy | window success ratio | window attempts / request |
|---|---|---|
| none | 0.750 [0.744, 0.755] | 1.000 [1.000, 1.000] |
| fixed2 | 1.000 [1.000, 1.000] | 1.250 [1.245, 1.256] |
| static alpha=0 | 0.856 [0.850, 0.861] | 1.106 [1.106, 1.107] |
| static alpha=0.8 | 1.000 [1.000, 1.000] | 1.250 [1.245, 1.256] |
| headroom alpha=0 | 1.000 [1.000, 1.000] | 1.250 [1.245, 1.256] |
| headroom alpha=0.8 | 1.000 [1.000, 1.000] | 1.250 [1.245, 1.256] |

With a static 10% budget, pricing every retry at one credit starves the failover
(0.856); pricing a warm retry at `1 - alpha * hit_ratio` credits (alpha = 0.8, the share a
cache hit saves) lets it through (1.000) with the same attempts per request as fixed
retries. The headroom budget also reaches 1.000 here without alpha, because at 150 rps
there is plenty of headroom. `alpha` still defaults to 0: the saving is a property of
the engine (compute-bound self-hosted pools such as vLLM or SGLang), and whether cached
tokens count toward a hosted provider's token quota (OpenAI, Anthropic, Groq) was not
measured. Set `INFERMESH_RETRY_ALPHA` for self-hosted pools.

An earlier version of this experiment let the outage hit all four servers; nothing could
warm a fallback during the outage, so `alpha` looked useless. That was the experiment's
bug, not a finding.

Run `uv run python -m bench.retry.run`; the output is deterministic and goes to the gitignored `bench/results/`.

## Real engines and real processes (2026-11-03)

The sections above are simulations. These runs use real processes on one Apple M4
(24 GB): llama.cpp `llama-server` 0.8.0 as the engine, k6 2.3.0 as the load tool,
the gateway under uvicorn. Metric choices follow the published setups: hit ratio,
throughput and latency percentiles for routing (Preble, SGLang v0.4, llm-d); added
latency against a fixed-latency upstream for overhead (LiteLLM's published method);
success rate, time to recover and attempts per request for faults (Bronson et al.,
SRE book). One machine runs engines, gateway and load generator together, so
absolute numbers are only comparable between arms of the same run. The model is a
small GGUF served from a local Ollama blob; its exact name was not recorded.

### Routing with 4 engines

8 prefix groups of about 300 words, 48 requests, concurrency 4, 2 KV slots per
engine, engine host cache disabled (`--cache-ram 0`) and slots flushed between arms
so arms cannot warm each other. Hit ratio is the engine's own `cached_tokens`.
3 repetitions per arm, arm order rotated; values are the three reps.

| arm | hit ratio | throughput (req/s) | p50 latency (s) |
|---|---|---|---|
| load only (cache threshold 1.0) | 0.20 0.24 0.12 | 0.27 0.28 0.25 | 13.7 14.0 15.4 |
| cache-aware, insert on completion | 0.36 0.30 0.54 | 0.33 0.30 0.44 | 13.1 13.5 7.0 |
| cache-aware, insert on dispatch | 0.60 0.42 0.42 | 0.51 0.37 0.37 | 6.1 11.7 11.7 |

A separate 3-rep run of load-only against cache-aware (insert on completion) gave
hit ratio 0.18 to 0.20 against 0.44 to 0.58 and throughput 0.24 to 0.27 against
0.27 to 0.48 req/s. Cache-aware routing beat load-only in every pair; that is the
consistent result (llm-d and SGLang report the same direction). Insert on dispatch
(`INFERMESH_INSERT_ON_DISPATCH`, default true, SGLang inserts at selection time)
beat insert on completion in two reps and lost in one (hit ratio 0.60/0.42/0.42 against 0.36/0.30/0.54); three reps do not show it is better, so it is
a default chosen to match SGLang, not a measured win. The low-sharing control
workload (`--kinds unique`) was not run.

### Gateway overhead

Fixed 50 ms upstream, k6 constant arrival rate, 30 s per point, one gateway
process, one worker. Direct is the same load without the gateway.

| concurrency cap | offered rps | direct p50 / p95 / p99 (ms) | gateway p50 / p95 / p99 (ms) | achieved rps direct / gateway |
|---|---|---|---|---|
| 20 | 100 | 51.0 / 57.4 / 58.3 | 53.3 / 62.0 / 98.9 | 100 / 100 |
| 20 | 200 | 50.3 / 411 / 646 | 52.8 / 1149 / 1241 | 199 / 199 |
| 20 | 300 | 50.3 / 1377 / 1642 | 78.0 / 2400 / 2636 | 299 / 287 |
| 1000 | 100 | 50.5 / 57.3 / 58.1 | 52.8 / 61.5 / 91.4 | 100 / 100 |
| 1000 | 300 | 50.3 / 773 / 1588 | 353 / 13806 / 16895 | 299 / 94 |
| 1000 | 600 | 50.3 / 51.2 / 2403 | 6188 / 14318 / 15809 | 594 / 59 |

At 100 rps the gateway adds 2.3 ms at p50 and 4.2 to 4.6 ms at p95, close to
LiteLLM's published 2 ms and 8 ms (different hardware, mock upstream in both). The
p99 addition (33 to 40 ms) is worse than LiteLLM's 13 ms. At 200 rps and above the
laptop is saturated and tails are poor for direct too, so those rows show the machine
more than the gateway. With the cap at 1000 the gateway collapses (goodput 94 and 59
rps): each request makes httpcore scan every open connection, and a high cap lets
about 1000 stay open. cProfile put the gateway at about 91% CPU, and an in-process
micro-benchmark gave about 0.48 ms per request, a ceiling near 2,100 rps per core.
Keep `max_concurrency_per_worker` modest; the default of 20 does not collapse.

### Fault injection

Three fake upstreams, open-loop Poisson load at 60 rps for 75 s with unique
prompts, one fault from 10 s to 22 s, one repetition per cell.

| scenario | arm | success inside the fault window | time to recover | upstream attempts per request |
|---|---|---|---|---|
| kill one worker | no retry budget | 100% | 0 s | 0.87 |
| kill one worker | static 10% budget | 100% | 0 s | 1.00 |
| all upstreams 503 | no retry budget | 0.4% | 19 s | 0.60 |
| all upstreams 503 | static 10% budget | 0.1% | 19 s | 0.60 |

The breaker opens within 0.2 s of the fault. After a full outage recovery takes 19 s
because the breaker's recovery timeout is 30 s. Attempts per request are below 1
because an open breaker answers without contacting any upstream. The budget made no
difference here: with a fixed 50 ms upstream there is no timeout-driven retry loop,
which the simulated E1 section above is built to produce. A 429 storm and a slow
worker were also tried and dropped: with one repetition the arms contradicted each
other, so the load was not reliably reaching the faulted worker.

```bash
MODEL_GGUF=/path/to/model.gguf uv run python -m bench.e2e.routing --out routing.json
uv run python -m bench.e2e.overhead --rates 100 200 300 --max-concurrency 20 --out overhead.json
uv run python -m bench.e2e.chaos --scenarios kill_worker total_outage --rate 60 --duration 75 --out chaos.json
```
