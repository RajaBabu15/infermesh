"""A small OpenAI-compatible HTTP server with a fixed latency and injectable faults.

  python -m bench.e2e.fake_upstream --port 9101 --latency-ms 50

Faults are switched at runtime: POST /__fault?mode=ok|status&code=429&retry_after=5|delay&ms=2000.
GET /__stats returns the number of chat requests received, so attempts per request can be counted.
"""
from __future__ import annotations

import argparse
import asyncio

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

STATE = {"mode": "ok", "code": 503, "retry_after": None, "delay_ms": 0, "latency_ms": 50, "received": 0}


async def chat(request):
    STATE["received"] += 1
    await request.body()
    if STATE["mode"] == "delay":
        await asyncio.sleep(STATE["delay_ms"] / 1000)
    elif STATE["mode"] == "status":
        headers = {"Retry-After": str(STATE["retry_after"])} if STATE["retry_after"] else {}
        return JSONResponse({"error": {"message": "injected"}}, status_code=STATE["code"], headers=headers)
    await asyncio.sleep(STATE["latency_ms"] / 1000)
    return JSONResponse({"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
                         "usage": {"prompt_tokens": 20, "completion_tokens": 2,
                                   "prompt_tokens_details": {"cached_tokens": 0}}})


async def fault(request):
    q = request.query_params
    STATE["mode"] = q.get("mode", "ok")
    STATE["code"] = int(q.get("code", 503))
    STATE["retry_after"] = int(q["retry_after"]) if "retry_after" in q else None
    STATE["delay_ms"] = int(q.get("ms", 0))
    return JSONResponse(dict(STATE))


async def stats(request):
    return JSONResponse({"received": STATE["received"]})


async def health(request):
    return JSONResponse({"status": "ok"})


app = Starlette(routes=[
    Route("/v1/chat/completions", chat, methods=["POST"]),
    Route("/__fault", fault, methods=["POST"]),
    Route("/__stats", stats),
    Route("/health", health),
])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--latency-ms", type=int, default=50)
    args = ap.parse_args()
    STATE["latency_ms"] = args.latency_ms
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="error")
