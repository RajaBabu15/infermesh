"""Gateway overhead: the same open-loop load sent directly to a fixed-latency upstream and
through the gateway, with k6 as the load generator (the method LiteLLM documents).

  uv run python -m bench.e2e.overhead --rates 100 300 600 --duration 30

Overhead at a rate is the gateway's latency percentile minus the direct one.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from bench.e2e.routing import wait_http

UPSTREAM_PORT, GATEWAY_PORT = 9101, 8766

K6_SCRIPT = """
import http from 'k6/http';
export const options = {
  scenarios: { load: { executor: 'constant-arrival-rate', rate: __RATE__, timeUnit: '1s',
    duration: '__DURATION__s', preAllocatedVUs: __VUS__, maxVUs: __VUS__ } },
  summaryTrendStats: ['p(50)', 'p(95)', 'p(99)', 'max', 'avg'],
};
const body = JSON.stringify({model: 'm', max_tokens: 4,
  messages: [{role: 'system', content: 'You are a helpful assistant.'}, {role: 'user', content: 'hi'}]});
export default function () {
  http.post('__URL__', body, {headers: {'Content-Type': 'application/json'}});
}
"""


def run_k6(url: str, rate: int, duration: int, vus: int) -> dict:
    script = K6_SCRIPT.replace("__RATE__", str(rate)).replace("__DURATION__", str(duration)) \
        .replace("__VUS__", str(vus)).replace("__URL__", url)
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "s.js").write_text(script)
        subprocess.run(["k6", "run", "--quiet", "--summary-export", f"{d}/out.json", f"{d}/s.js"],
                       check=True, capture_output=True)
        m = json.load(open(f"{d}/out.json"))["metrics"]
    dur = m["http_req_duration"]
    return {"rate": rate, "achieved_rps": m["http_reqs"]["rate"], "p50_ms": dur["p(50)"], "p95_ms": dur["p(95)"],
            "p99_ms": dur["p(99)"], "avg_ms": dur["avg"], "max_ms": dur["max"],
            "error_rate": m.get("http_req_failed", {}).get("value", 0.0),
            "dropped": m.get("dropped_iterations", {}).get("count", 0)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rates", type=int, nargs="+", default=[100, 300, 600])
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--latency-ms", type=int, default=50)
    ap.add_argument("--vus", type=int, default=400)
    ap.add_argument("--max-concurrency", type=int, default=20, help="per-worker concurrency cap of the gateway")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp())
    (tmp / "workers.yaml").write_text(
        f"workers:\n  - {{id: up, type: openai, url: 'http://127.0.0.1:{UPSTREAM_PORT}/v1', model: m, api_key_env: NO_KEY}}\n")
    upstream = subprocess.Popen([sys.executable, "-m", "bench.e2e.fake_upstream", "--port", str(UPSTREAM_PORT),
                                 "--latency-ms", str(args.latency_ms)])
    env = {**os.environ, "INFERMESH_WORKERS_CONFIG_PATH": str(tmp / "workers.yaml"), "INFERMESH_LOG_LEVEL": "WARNING",
           "INFERMESH_MAX_CONCURRENCY_PER_WORKER": str(args.max_concurrency), "INFERMESH_MAX_QUEUE_PER_WORKER": "1000"}
    gateway = subprocess.Popen([sys.executable, "-m", "uvicorn", "infermesh.main:create_app", "--factory",
                                "--port", str(GATEWAY_PORT), "--log-level", "warning"], env=env)
    rows = []
    try:
        wait_http(f"http://127.0.0.1:{UPSTREAM_PORT}/health")
        wait_http(f"http://127.0.0.1:{GATEWAY_PORT}/health")
        run_k6(f"http://127.0.0.1:{UPSTREAM_PORT}/v1/chat/completions", 50, 5, args.vus)   # warm-up
        run_k6(f"http://127.0.0.1:{GATEWAY_PORT}/v1/chat/completions", 50, 5, args.vus)
        for rate in args.rates:
            direct = run_k6(f"http://127.0.0.1:{UPSTREAM_PORT}/v1/chat/completions", rate, args.duration, args.vus)
            via = run_k6(f"http://127.0.0.1:{GATEWAY_PORT}/v1/chat/completions", rate, args.duration, args.vus)
            row = {"rate": rate, "direct": direct, "gateway": via,
                   "overhead_p50_ms": via["p50_ms"] - direct["p50_ms"],
                   "overhead_p95_ms": via["p95_ms"] - direct["p95_ms"],
                   "overhead_p99_ms": via["p99_ms"] - direct["p99_ms"]}
            rows.append(row)
            print(json.dumps({"rate": rate, "direct_p50/p95/p99": [round(direct[k], 1) for k in ("p50_ms", "p95_ms", "p99_ms")],
                              "gateway_p50/p95/p99": [round(via[k], 1) for k in ("p50_ms", "p95_ms", "p99_ms")],
                              "overhead_p50/p95/p99": [round(row[k], 1) for k in ("overhead_p50_ms", "overhead_p95_ms", "overhead_p99_ms")],
                              "achieved_rps": [round(direct["achieved_rps"]), round(via["achieved_rps"])],
                              "errors": [direct["error_rate"], via["error_rate"]], "dropped": [direct["dropped"], via["dropped"]]}),
                  flush=True)
    finally:
        gateway.terminate(); upstream.terminate()
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
