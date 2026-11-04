"""Redis event dispatch + stream entry handling (at-least-once XACK)."""
import asyncio
import json

from conftest import DummyLog, cfg
from infermesh.kv_subscriber import SeqTracker, _dispatch_event, _handle_stream_entry
from infermesh.trie import RadixTrie


def test_dispatch_kv_cached_then_evicted():
    t = RadixTrie(ttl_s=1e9)
    _dispatch_event({"event": "kv_cached", "worker_id": "w", "model": "m", "prefix": "hello"},
                    cfg(), t, DummyLog())
    assert t.longest_prefix_match("m", "hello")[0] == ["w"]
    _dispatch_event({"event": "kv_evicted", "worker_id": "w", "model": "m", "prefix": "hello"},
                    cfg(), t, DummyLog())
    assert t.longest_prefix_match("m", "hello")[0] == []


def test_dispatch_incomplete_event_is_dropped():
    t = RadixTrie(ttl_s=1e9)
    _dispatch_event({"event": "kv_cached", "worker_id": "w", "model": "m"},
                    cfg(), t, DummyLog())
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
                                     cfg(), t, DummyLog()))
    assert t.longest_prefix_match("m", "hi")[0] == ["w"]
    assert fr.acked == ["1-0"]


def test_malformed_stream_entry_is_acked_not_stuck():
    fr = FakeRedis()
    asyncio.run(_handle_stream_entry(fr, "stream", "grp", "2-0", {"data": "notjson{"},
                                     cfg(), RadixTrie(ttl_s=1e9), DummyLog()))
    assert fr.acked == ["2-0"]


def _cached(worker, prefix, seq=None):
    event = {"event": "kv_cached", "worker_id": worker, "model": "m", "prefix": prefix}
    if seq is not None:
        event["seq"] = seq
    return event


def test_kv_cleared_drops_the_worker():
    t = RadixTrie(ttl_s=1e9)
    _dispatch_event(_cached("w1", "hello"), cfg(), t, DummyLog())
    _dispatch_event(_cached("w2", "hello"), cfg(), t, DummyLog())
    _dispatch_event({"event": "kv_cleared", "worker_id": "w1"}, cfg(), t, DummyLog())
    assert t.longest_prefix_match("m", "hello")[0] == ["w2"]
    _dispatch_event({"event": "kv_cleared"}, cfg(), t, DummyLog())       # no worker: ignored
    assert t.longest_prefix_match("m", "hello")[0] == ["w2"]


def test_seq_tracker_verdicts():
    s = SeqTracker()
    assert s.check("w", 5) == ("ok", 0)               # first seen
    assert s.check("w", 6) == ("ok", 0)
    assert s.check("w", 6) == ("duplicate", 0)
    assert s.check("w", 4) == ("duplicate", 0)
    assert s.check("w", 9) == ("gap", 2)
    assert s.check("w", 10) == ("ok", 0)
    assert s.check("w", 0) == ("restart", 0)
    assert s.check("other", 100) == ("ok", 0)         # independent per worker


def test_seq_gap_drops_stale_entries_and_applies_the_new_event():
    t, seq = RadixTrie(ttl_s=1e9), SeqTracker()
    _dispatch_event(_cached("w1", "old-prefix", seq=1), cfg(), t, DummyLog(), seq)
    _dispatch_event(_cached("w1", "new-prefix", seq=5), cfg(), t, DummyLog(), seq)   # 2,3,4 lost
    assert t.longest_prefix_match("m", "old-prefix")[0] == []         # unsafe to keep
    assert t.longest_prefix_match("m", "new-prefix")[0] == ["w1"]


def test_duplicate_event_is_skipped_so_a_late_replay_cannot_undo_a_newer_one():
    t, seq = RadixTrie(ttl_s=1e9), SeqTracker()
    _dispatch_event(_cached("w1", "p", seq=1), cfg(), t, DummyLog(), seq)
    _dispatch_event({"event": "kv_evicted", "worker_id": "w1", "model": "m", "prefix": "p", "seq": 2},
                    cfg(), t, DummyLog(), seq)
    _dispatch_event(_cached("w1", "p", seq=2), cfg(), t, DummyLog(), seq)   # replayed copy of seq 2
    assert t.longest_prefix_match("m", "p")[0] == []                  # not resurrected


def test_restart_event_clears_the_worker():
    t, seq = RadixTrie(ttl_s=1e9), SeqTracker()
    _dispatch_event(_cached("w1", "before", seq=7), cfg(), t, DummyLog(), seq)
    _dispatch_event(_cached("w1", "after", seq=0), cfg(), t, DummyLog(), seq)
    assert t.longest_prefix_match("m", "before")[0] == []
    assert t.longest_prefix_match("m", "after")[0] == ["w1"]


def test_seq_state_is_not_shared_between_trackers_or_loops():
    import infermesh.kv_subscriber as kv

    assert not hasattr(kv, "_seq_tracker")                                  # no module-level state
    a, b = SeqTracker(), SeqTracker()
    assert a.check("w1", 5) == ("ok", 0) and a.check("w1", 5) == ("duplicate", 0)
    assert b.check("w1", 5) == ("ok", 0)                                    # a fresh tracker knows nothing
    t = RadixTrie(ttl_s=1e9)                                               # and without a tracker, seq is ignored
    for _ in range(2):
        _dispatch_event(_cached("w1", "p", seq=5), cfg(), t, DummyLog())
    assert t.longest_prefix_match("m", "p")[0] == ["w1"]
