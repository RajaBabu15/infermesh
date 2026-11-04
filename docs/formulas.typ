#set page(paper: "a4", margin: (x: 2.3cm, y: 2.1cm))
#set text(size: 10.5pt)
#set par(justify: true, leading: 0.68em)
#set heading(numbering: "1.")
#show heading: set text(size: 11pt, weight: "semibold")
#show heading: set block(above: 1.15em, below: 0.45em)
#show raw: set text(size: 9pt)
#show raw: it => box(text(hyphenate: false, it))
#show table: set par(justify: false, leading: 0.62em)

#align(center)[
  #text(15pt, weight: "bold")[Formulas]
  #v(0.35em)
  #text(9.5pt)[InferMesh gateway. Defaults unless named. _Assumed_ means used and unpublished. Simulator-only identities at the end.]
]

#v(0.6em)

Notation. Pool $W$ for a model: breaker admits the worker and waiters $< Q$. Load $L(w)=$ requests in flight. Prompt string $p$. Match depth $d$. Attempts $A=2$.

= Prefix

$
p = "join"(sans("role"): sans("content"))
$

over `messages`, newline-separated (`RadixTrie.extract_prefix`). Completions use the raw `prompt` string. Embeddings skip the trie.

= Token reserve (ETIF)

Heuristic, not a tokenizer. Let $c$ be the sum of `content` lengths in `messages` (roles are not counted).

$
hat(n) = max(1, lr(floor.l c/4 floor.r)) + #raw("max_tokens")
$

(or $256$ if the client omits `max_tokens`). Set `etif_count_output_tokens=false` to drop the second term. Reserved until the response ends. $c$ is not $|p|$: the trie string includes `role: `.

= Selection

*Problem.* Send $p$ to a worker that already holds a long prefix of $p$, unless that pins load.

*SGLang cache-aware* (`sgl-model-gateway`). CLI defaults: cache-threshold 0.3, balance-abs-threshold 64, balance-rel-threshold 1.5. Load is in-flight count. Imbalanced iff both

$
max L - min L > 64
quad "and" quad
max L > 1.5 min L.
$

Otherwise a hit iff there exist live tenants of the longest match with

$
d >= #raw("kv_match_min_chars")
quad "and" quad
d > 0.3 |p|.
$

Then pick $op("arg min") L$ among those tenants.

*Here.* Same predicates. On miss or imbalance: power-of-two choices, not SGLang's min-load (random tie-break). Match rate in SGLang is $d\/|p|$ in characters (`prefix_match_with_counts`). Same here.

Insert of $p$ on dispatch (`insert_on_dispatch=true`), as SGLang inserts at `select_worker`.

= Power of two choices

*Theorem* (Azar–Broder–Karlin–Upfal, SIAM J. Comput. 29 (1999)). $n$ balls, $n$ bins, each ball into the least loaded of $d >= 2$ uniform bins: max load $(log log n)\/(log d) + O(1)$ w.h.p. For $d=1$ it is $Theta(log n \/ log log n)$.

*Supermarket* (Mitzenmacher, IEEE TPDS 12 (2001)). Poisson arrivals rate $lambda n$, $lambda < 1$, exponential service; $d=2$ cuts expected wait exponentially vs $d=1$.

*Here.* $d=2$. Sample without replacement, weights $=$ worker `weight`. Compare scores, not raw queue length. The theorem applies only while the score is a live queue; it does not cover a stale or mixed score.

= Score

Let $T(w)$ be tokens in flight, $U(w) in [0,1]$ engine KV util, $q(w)$ engine wait depth.

$
s(w) = cases(
  min(T(w)\/S, 1)
    & "if openai or " U " stale",
  alpha min(T\/S, 1) + beta min(q\/q_0, 1) + (1-alpha-beta) U
    & "otherwise",
)
$

Defaults: $alpha=0.6$, $beta=0$, $q_0=8$, $S=20 dot 1024$ (or `etif_scale`). Stale means older than $3$ poll intervals. $alpha=0.6$ is _assumed_ (no published mix). With $beta=0$ the vLLM term is $0.4 U$.

= Trie

Character radix (Morrison PATRICIA, 1968): one child per first character, split on LCP. Tenants stamped on every node of an insert. Match may stop inside an edge.

TTL $300 thin "s"$ per attribution. Cap $2^(26)$ characters per worker (SGLang `max_tree_size`). Over cap: drop LRU leaves until $0.9$ of the cap (_assumed_ headroom; SGLang uses a periodic node LRU).

LCP length: binary search on slices, $O(log n)$ Python steps, $O(n)$ bytes compared.

= Retry budget

*SRE* (_Site Reliability Engineering_, Ch. 21): retry while the retry ratio stays below 10 percent; amplification at most about $1.1$ vs about $3$ with a per-request cap of three.

*Envoy* `RetryBudget`: default 20 percent of active+pending, floor `min_retry_concurrency=3`.

*Here.* Credit ledger, mode `static`:

$
c <- min(c_"max", c + 0.10)
quad "per client request"
$

$
rho = rho_0 (1 - alpha_rho h),
quad rho_0 = 1, thick alpha_rho = 0.
$

Admit a retry iff $c >= rho$. Then $c <- c - rho$. With $alpha_rho=0$, $rho=1$: one retry per ten requests in the mean, burst $c_0=3$, cap $c_"max"=10$. $h = d_w \/ |p|$ for the fallback worker; unused while $alpha_rho=0$.

Mode `headroom` (_assumed_, off): deposit $max(0, min(R\/(rho_0 lambda)-1, 1))$ once $lambda$ is known over a window of at least $1 thin "s"$. Local reroutes (open breaker, full queue) skip the ledger.

Hard cap $A=2$ attempts, distinct workers. Retry only before the first byte, on 429, 5xx, or transport. Timeouts are not retried.

= Circuit breaker

*Envoy outlier.* `consecutive_5xx=5`. Real ejection time $=$ `base_ejection_time` times consecutive ejections, capped by `max_ejection_time`. Defaults $30 thin "s"$ and $300 thin "s"$.

$
T(k) = min(T_0 k, T_"max"),
quad T_0 = 30, thick T_"max" = 300.
$

First open $k=1 => T=30 thin "s"$. Success sets $k=0$. Explicit `Retry-After` cool-down does not increment $k$.

States: CLOSED $->$ OPEN after 5 consecutive 5xx (not 429) $->$ HALF_OPEN after $T$, one probe. 429 opens for the header delay, unmultiplied.

Not taken from Envoy: `max_ejection_percent` (default 10 percent); the interval that decrements the multiplier while healthy (here a success zeros $k$).

= Cooldown headers

*RFC 9110 §10.2.3.* `Retry-After = HTTP-date / delay-seconds`. Invalid $=>$ ignore.

Else the largest vendor reset (`x-ratelimit-reset-*`, Anthropic RFC 3339 / epoch). Cap $300 thin "s"$. Never $infinity$.

= Queue

Semaphore $C=20$ in-flight per worker. Waiters $<= Q=40$, wait $<= 30 thin "s"$, else 503 with `Retry-After`. Routable while waiters $< Q$.

= Replay / retry benches (not the gateway)

Prefill time, DistServe (arXiv 2401.09670) Appendix A _form_ $T = C_1(dots.c) + C_2(dots.c) + C_3$, constants fitted per model/GPU and unpublished:

$
t_"prefill"(u) = c_1 u + c_2 u^2 + c_3
$

with $c_1 = 4 dot 10^(-5)$, $c_2 = 3 dot 10^(-9)$, $c_3 = 0.02$, ITL $0.02 thin "s"$ _assumed_. Claim policy order, not seconds.

SLO in `bench/replay/metrics.py`: TTFT $0.25 thin "s"$, TPOT $0.10 thin "s"$ (DistServe Table 1, OPT-13B chatbot). `docs/benchmarks.md` reports a $2 thin "s"$ TTFT cut on the long Mooncake prompts.

Retry sim: Bronson et al., HotOS 2021 §2.1, capacity 300 QPS, offered 280 QPS.

= Sources

#table(
  columns: (1.35fr, 2.1fr, 1.7fr),
  stroke: 0.4pt,
  inset: 6pt,
  [*Object*], [*Statement*], [*Here*],
  [Cache hit / imbalance], [SGLang CLI `router_args.py` $0.3\/64\/1.5$; `cache_aware.rs`], [same predicates; miss $->$ P2C],
  [$d=2$ max load], [Azar et al. 1999], [score, not queue],
  [Retry ratio 10 percent], [SRE Ch. 21], [deposit $0.10$, $rho=1$],
  [$T = T_0 k$], [Envoy outlier], [`proxy.py` `CircuitBreaker._open`],
  [`Retry-After`], [RFC 9110 §10.2.3], [`ratelimit.py`],
  [Prefill $C_i$], [DistServe App. A], [replay only, $C_i$ assumed],
  [$alpha=0.6$, $alpha_rho$, headroom $0.9$], [—], [_assumed_],
)

#v(1em)
#text(8.5pt)[Compile: `typst compile docs/formulas.typ docs/formulas.pdf`]
