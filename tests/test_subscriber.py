"""Redis event dispatch + stream entry handling (at-least-once XACK)."""
import asyncio
import json

from infermesh.kv_subscriber import _dispatch_event, _handle_stream_entry
from infermesh.routing import RadixTrie, SessionTracker
from infermesh.settings import GatewayConfig


class DummyLog:
    def __getattr__(self, name):
        return lambda *a, **k: None


def cfg():
    return GatewayConfig(_env_file=None)


def test_dispatch_kv_cached_then_evicted():
    t = RadixTrie(ttl_s=1e9)
    _dispatch_event({"event": "kv_cached", "worker_id": "w", "model": "m", "prefix": "hello"},
                    cfg(), t, None, DummyLog())
    assert t.longest_prefix_match("m", "hello")[0] == "w"
    _dispatch_event({"event": "kv_evicted", "worker_id": "w", "model": "m", "prefix": "hello"},
                    cfg(), t, None, DummyLog())
    assert t.longest_prefix_match("m", "hello")[0] is None


def test_dispatch_prefill_complete_registers_session():
    sessions = SessionTracker(ttl_s=1e9)
    _dispatch_event({"event": "prefill_complete", "session_id": "s",
                     "decode_worker_id": "d", "model": "m"},
                    cfg(), RadixTrie(ttl_s=1e9), sessions, DummyLog())
    assert sessions.lookup("s") == ("d", "m")
    assert sessions.prefill_complete_received_total == 1


def test_dispatch_incomplete_event_is_dropped():
    t = RadixTrie(ttl_s=1e9)
    # missing prefix -> dropped, no crash, trie unchanged
    _dispatch_event({"event": "kv_cached", "worker_id": "w", "model": "m"},
                    cfg(), t, None, DummyLog())
    assert t.entry_count() == 0


class FakeRedis:
    def __init__(self):
        self.acked = []

    async def xack(self, stream, group, entry_id):
        self.acked.append(entry_id)


def test_stream_entry_dispatched_and_acked():
    t = RadixTrie(ttl_s=1e9)
    fr = FakeRedis()
    fields = {"data": json.dumps({"event": "kv_cached", "worker_id": "w",
                                  "model": "m", "prefix": "hi"})}
    asyncio.run(_handle_stream_entry(fr, "stream", "grp", "1-0", fields,
                                     cfg(), t, None, DummyLog()))
    assert t.longest_prefix_match("m", "hi")[0] == "w"
    assert fr.acked == ["1-0"]


def test_malformed_stream_entry_is_acked_not_stuck():
    """A bad-JSON entry must still be XACKed so it doesn't wedge the PEL."""
    fr = FakeRedis()
    asyncio.run(_handle_stream_entry(fr, "stream", "grp", "2-0", {"data": "notjson{"},
                                     cfg(), RadixTrie(ttl_s=1e9), None, DummyLog()))
    assert fr.acked == ["2-0"]
