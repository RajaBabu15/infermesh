"""Optional Redis consumer for worker-published prefix events.

Transport (`redis_transport`): `streams` is at-least-once with replay, `pubsub`
is fire-and-forget. Payloads are JSON; streams wraps them in a `data` field.

Events: `kv_cached` and `kv_evicted` ({worker_id, model, prefix}), and
`kv_cleared` ({worker_id}, an engine cache flush). An optional per-worker `seq`
detects loss: a gap or a restart (seq 0) drops that worker's entries so the router
load-balances instead of trusting a stale index, and seq <= last is a replayed
duplicate and is skipped.

No stock engine publishes this schema; a sidecar has to.
"""
from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import structlog

from infermesh import metrics
from infermesh.tracing import get_tracer

if TYPE_CHECKING:
    from infermesh.trie import RadixTrie
    from infermesh.settings import GatewayConfig

_BACKOFF_BASE_S: float = 1.0
_BACKOFF_MAX_S: float = 60.0
_STREAMS_BLOCK_MS: int = 5000
# Must exceed the XREADGROUP block time, or every idle read raises
# TimeoutError (redis-py 8 defaults socket_timeout to 5 s; redis-py #2807).
_SOCKET_TIMEOUT_S: float = _STREAMS_BLOCK_MS / 1000 + 5
_STREAMS_BATCH_COUNT: int = 64

_tracer = get_tracer(__name__)


class SeqTracker:
    """Per-worker event sequence numbers: detect gaps, duplicates and restarts."""

    def __init__(self) -> None:
        self._last: dict[str, int] = {}

    def check(self, worker_id: str, seq: int) -> tuple[str, int]:
        """Return (verdict, missed): ok, duplicate, gap or restart."""
        last = self._last.get(worker_id)
        if last is None:
            self._last[worker_id] = seq
            return "ok", 0
        if seq == 0 and last > 0:
            self._last[worker_id] = 0
            return "restart", 0
        if seq <= last:
            return "duplicate", 0
        self._last[worker_id] = seq
        if seq == last + 1:
            return "ok", 0
        return "gap", seq - last - 1


async def kv_subscriber_loop(
    trie: "RadixTrie",
    config: "GatewayConfig",
) -> None:
    """Long-running task that drains Redis events into trie and sessions."""
    import redis.asyncio as aioredis

    log = structlog.get_logger()
    delay = _BACKOFF_BASE_S
    # One tracker per loop: it survives Redis reconnects (so a gap or a replayed
    # duplicate across a reconnect is still seen) but not a new app or test.
    seq_tracker = SeqTracker()

    while True:
        client: aioredis.Redis | None = None
        try:
            client = aioredis.from_url(
                config.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=_SOCKET_TIMEOUT_S,
            )
            await client.ping()
            delay = _BACKOFF_BASE_S
            if config.redis_transport == "streams":
                await _consume_streams(client, config, trie, log, seq_tracker)
            else:
                await _consume_pubsub(client, config, trie, log, seq_tracker)

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
                except Exception:  # connection is already broken; the reconnect path matters, not this close
                    pass

        await asyncio.sleep(delay)
        delay = min(delay * 2, _BACKOFF_MAX_S)


async def _consume_streams(
    client: Any,
    config: "GatewayConfig",
    trie: "RadixTrie",
    log: Any,
    seq_tracker: SeqTracker | None = None,
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
                client, stream_key, group, entry_id, fields, config, trie, log,
                seq_tracker,
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
                    client, stream_key, group, entry_id, fields, config, trie, log,
                    seq_tracker,
                )


async def _handle_stream_entry(
    client: Any,
    stream_key: str,
    group: str,
    entry_id: str,
    fields: dict,
    config: "GatewayConfig",
    trie: "RadixTrie",
    log: Any,
    seq_tracker: SeqTracker | None = None,
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
        _dispatch_event(event, config, trie, log, seq_tracker)
    finally:
        await client.xack(stream_key, group, entry_id)


async def _consume_pubsub(
    client: Any,
    config: "GatewayConfig",
    trie: "RadixTrie",
    log: Any,
    seq_tracker: SeqTracker | None = None,
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
        _dispatch_event(event, config, trie, log, seq_tracker)


def _dispatch_event(
    event: dict,
    config: "GatewayConfig",
    trie: "RadixTrie",
    log: Any,
    seq_tracker: SeqTracker | None = None,
) -> None:
    event_type = event.get("event", "")
    model = event.get("model", "")
    metrics.redis_events_received_total.labels(event_type=event_type or "unknown").inc()

    seq = event.get("seq")
    event_worker = event.get("worker_id", "")
    if seq_tracker is not None and isinstance(seq, int) and not isinstance(seq, bool) and event_worker:
        verdict, missed = seq_tracker.check(event_worker, seq)
        if verdict == "duplicate":
            metrics.redis_events_dropped_total.labels(reason="duplicate").inc()
            return
        if verdict in ("gap", "restart"):
            dropped = trie.remove_worker(event_worker)
            if missed:
                metrics.redis_event_seq_gaps_total.labels(worker_id=event_worker).inc(missed)
            log.warning("kv_subscriber_resync", worker_id=event_worker, verdict=verdict,
                        missed=missed, entries_dropped=dropped)

    if event_type == "kv_cleared":
        if event_worker:
            trie.remove_worker(event_worker)
            log.debug("kv_cleared", worker_id=event_worker)
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
            found = trie.remove(model, prefix, worker_id)
            log.debug("kv_evicted", worker_id=worker_id, model=model, found=found)
    else:
        log.warning("kv_subscriber_unknown_event", event_type=event_type)
        metrics.redis_events_dropped_total.labels(reason="unknown").inc()
