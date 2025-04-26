"""Reproducible demonstration of at-least-once Redis Streams replay.

Shows that events delivered to a consumer but never XACK'd (e.g. the gateway
crashed mid-batch) are replayed on restart via the pending-entries-list drain
in infermesh.kv_subscriber._consume_streams (phase 1, id "0").

Run (needs a local Redis):
    redis-server --port 6379 --save "" &
    uv run python bench/redis_replay_demo.py
"""
from __future__ import annotations

import asyncio
import json

import redis.asyncio as aioredis

from infermesh.kv_subscriber import _handle_stream_entry
from infermesh.routing import RadixTrie
from infermesh.settings import GatewayConfig

STREAM = "infermesh:replay_demo"
GROUP = "infermesh-gateway"
CONSUMER = "gateway-0"
N = 5


class _Log:
    def __getattr__(self, _):
        return lambda *a, **k: None


async def main() -> None:
    r = aioredis.from_url("redis://127.0.0.1:6379", decode_responses=True)
    config = GatewayConfig(_env_file=None)
    log = _Log()
    await r.delete(STREAM)
    try:
        await r.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
    except Exception as exc:
        if "BUSYGROUP" not in str(exc):
            raise

    # Publish N kv_cached events.
    for i in range(N):
        await r.xadd(STREAM, {"data": json.dumps(
            {"event": "kv_cached", "worker_id": "w", "model": "m", "prefix": f"prefix-{i}"})})

    # Simulate a consumer that READ the batch (now PENDING) but crashed before
    # processing/acking — i.e. deliver to the group without _handle_stream_entry.
    await r.xreadgroup(GROUP, CONSUMER, {STREAM: ">"}, count=N)
    pending_before = (await r.xpending(STREAM, GROUP))["pending"]

    # "Restart": drain the pending-entries list (id "0") exactly as
    # _consume_streams phase 1 does, into a fresh trie.
    trie = RadixTrie(ttl_s=1e9)
    resp = await r.xreadgroup(GROUP, CONSUMER, {STREAM: "0"}, count=N)
    pending = [e for _k, entries in (resp or []) for e in entries]
    for entry_id, fields in pending:
        await _handle_stream_entry(r, STREAM, GROUP, entry_id, fields, config, trie, None, log)

    pending_after = (await r.xpending(STREAM, GROUP))["pending"]
    recovered = sum(1 for i in range(N)
                    if trie.longest_prefix_match("m", f"prefix-{i}")[0] == "w")

    print(f"pending before restart : {pending_before} (un-acked, would be lost reading '>')")
    print(f"replayed into trie      : {recovered}/{N}")
    print(f"pending after drain     : {pending_after}")
    ok = pending_before == N and recovered == N and pending_after == 0
    print(f"[{'PASS' if ok else 'FAIL'}] at-least-once replay of un-acked entries")
    await r.delete(STREAM)
    await r.aclose()


if __name__ == "__main__":
    asyncio.run(main())
