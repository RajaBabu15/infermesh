"""Redis subscriber that feeds worker-reported KV state into the trie.

Transports (chosen via GatewayConfig.redis_transport):
  streams: Redis Streams + consumer groups, at-least-once with replay.
  pubsub:  fire-and-forget; lost messages cannot be replayed.

Payloads are JSON in both transports. Streams wraps the JSON in a single
{"data": "<json>"} field per entry.

Event types:
  kv_cached:        {"event", "worker_id", "model", "prefix"}
  kv_evicted:       {"event", "worker_id", "model", "prefix"}
  prefill_complete: {"event", "session_id", "prefill_worker_id",
                     "decode_worker_id", "model", "prefix_len", "kv_size_mb"}

Graceful degradation:
  INFERMESH_REDIS_URL="": loop never starts.
  Redis drops:            reconnect with exponential backoff.
  sessions=None:          prefill_complete events silently dropped.
"""
from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import structlog

from infermesh import metrics
from infermesh.tracing import get_tracer

if TYPE_CHECKING:
    from infermesh.routing import RadixTrie, SessionTracker
    from infermesh.settings import GatewayConfig

_BACKOFF_BASE_S: float = 1.0
_BACKOFF_MAX_S: float = 60.0
_STREAMS_BLOCK_MS: int = 5000
_STREAMS_BATCH_COUNT: int = 64

_tracer = get_tracer(__name__)


async def kv_subscriber_loop(
    trie: "RadixTrie",
    config: "GatewayConfig",
    sessions: "SessionTracker | None" = None,
) -> None:
    """Long-running task that drains Redis events into trie and sessions."""
    import redis.asyncio as aioredis

    log = structlog.get_logger()
    delay = _BACKOFF_BASE_S

    while True:
        client: aioredis.Redis | None = None
        try:
            client = aioredis.from_url(
                config.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            if config.redis_transport == "streams":
                await _consume_streams(client, config, trie, sessions, log)
            else:
                await _consume_pubsub(client, config, trie, sessions, log)
            delay = _BACKOFF_BASE_S

        except asyncio.CancelledError:
            log.info("kv_subscriber_cancelled")
            break

        except Exception as exc:
            log.warning(
                "kv_subscriber_connection_error",
                error=str(exc),
                transport=config.redis_transport,
                reconnect_in_s=delay,
            )

        finally:
            if client is not None:
                try:
                    await client.aclose()
                except Exception:
                    pass

        await asyncio.sleep(delay)
        delay = min(delay * 2, _BACKOFF_MAX_S)


# ---------------------------------------------------------------------------
# Streams transport
# ---------------------------------------------------------------------------

async def _consume_streams(
    client: Any,
    config: "GatewayConfig",
    trie: "RadixTrie",
    sessions: "SessionTracker | None",
    log: Any,
) -> None:
    """XREADGROUP loop. Drains the PEL (id \"0\") before reading new entries (\">\")."""
    stream_key = config.redis_channel
    group = config.redis_stream_consumer_group
    consumer = config.redis_stream_consumer_name

    try:
        await client.xgroup_create(stream_key, group, id="0", mkstream=True)
    except Exception as exc:
        if "BUSYGROUP" not in str(exc):
            raise

    log.info(
        "kv_subscriber_streams_connected",
        stream=stream_key,
        group=group,
        consumer=consumer,
    )

    replayed = 0
    while True:
        resp = await client.xreadgroup(
            group, consumer, {stream_key: "0"}, count=_STREAMS_BATCH_COUNT,
        )
        pending = [e for _k, entries in (resp or []) for e in entries]
        if not pending:
            break
        for entry_id, fields in pending:
            await _handle_stream_entry(
                client, stream_key, group, entry_id, fields, config, trie, sessions, log
            )
            replayed += 1
    if replayed:
        log.info("kv_subscriber_replayed_pending", count=replayed)

    while True:
        resp = await client.xreadgroup(
            group,
            consumer,
            {stream_key: ">"},
            count=_STREAMS_BATCH_COUNT,
            block=_STREAMS_BLOCK_MS,
        )
        if not resp:
            continue
        for _stream_key, entries in resp:
            for entry_id, fields in entries:
                await _handle_stream_entry(
                    client, stream_key, group, entry_id, fields, config, trie, sessions, log
                )


async def _handle_stream_entry(
    client: Any,
    stream_key: str,
    group: str,
    entry_id: str,
    fields: dict,
    config: "GatewayConfig",
    trie: "RadixTrie",
    sessions: "SessionTracker | None",
    log: Any,
) -> None:
    """Parse + dispatch a single stream entry, XACKing it exactly once."""
    raw = fields.get("data", "")
    try:
        event = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        log.warning("kv_subscriber_bad_json", raw=repr(raw)[:200])
        metrics.redis_events_dropped_total.labels(reason="bad_json").inc()
        await client.xack(stream_key, group, entry_id)
        return
    try:
        _dispatch_event(event, config, trie, sessions, log)
    finally:
        await client.xack(stream_key, group, entry_id)


# ---------------------------------------------------------------------------
# Pub/Sub transport
# ---------------------------------------------------------------------------

async def _consume_pubsub(
    client: Any,
    config: "GatewayConfig",
    trie: "RadixTrie",
    sessions: "SessionTracker | None",
    log: Any,
) -> None:
    ps = client.pubsub()
    await ps.subscribe(config.redis_channel)
    log.info("kv_subscriber_pubsub_connected", channel=config.redis_channel)

    async for message in ps.listen():
        if message["type"] != "message":
            continue
        raw = message["data"]
        try:
            event = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            log.warning("kv_subscriber_bad_json", raw=repr(raw)[:200])
            metrics.redis_events_dropped_total.labels(reason="bad_json").inc()
            continue
        _dispatch_event(event, config, trie, sessions, log)


# ---------------------------------------------------------------------------
# Event dispatch
# ---------------------------------------------------------------------------

def _dispatch_event(
    event: dict,
    config: "GatewayConfig",
    trie: "RadixTrie",
    sessions: "SessionTracker | None",
    log: Any,
) -> None:
    event_type = event.get("event", "")
    model = event.get("model", "")
    metrics.redis_events_received_total.labels(event_type=event_type or "unknown").inc()

    if event_type == "prefill_complete":
        if sessions is None:
            return
        sid = event.get("session_id", "")
        decode_wid = event.get("decode_worker_id", "")
        if sid and decode_wid and model:
            with _tracer.start_as_current_span(
                "kv_event.prefill_complete",
                attributes={
                    "session.id": sid,
                    "decode_worker.id": decode_wid,
                    "model": model,
                },
            ):
                sessions.register(sid, decode_wid, model)
                log.debug(
                    "prefill_complete",
                    session_id=sid,
                    decode_worker=decode_wid,
                    prefix_len=event.get("prefix_len", 0),
                    kv_size_mb=event.get("kv_size_mb", 0.0),
                )
        else:
            log.warning("kv_subscriber_incomplete_event", event=str(event))
            metrics.redis_events_dropped_total.labels(reason="incomplete").inc()
        return

    worker_id = event.get("worker_id", "")
    prefix = event.get("prefix", "")
    if not (worker_id and model and prefix):
        log.warning("kv_subscriber_incomplete_event", event=str(event))
        metrics.redis_events_dropped_total.labels(reason="incomplete").inc()
        return

    if event_type == "kv_cached":
        with _tracer.start_as_current_span(
            "kv_event.kv_cached",
            attributes={
                "worker.id": worker_id,
                "model": model,
                "prefix_len": len(prefix),
            },
        ):
            trie.insert(model, prefix, worker_id, ttl_override_s=config.redis_event_ttl_s)
            log.debug("kv_cached", worker_id=worker_id, model=model, prefix_len=len(prefix))
    elif event_type == "kv_evicted":
        with _tracer.start_as_current_span(
            "kv_event.kv_evicted",
            attributes={"worker.id": worker_id, "model": model},
        ):
            found = trie.remove(model, prefix)
            log.debug("kv_evicted", worker_id=worker_id, model=model, found=found)
    else:
        log.warning("kv_subscriber_unknown_event", event_type=event_type)
        metrics.redis_events_dropped_total.labels(reason="unknown").inc()
