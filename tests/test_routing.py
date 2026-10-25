"""RadixTrie correctness (incl. a longest-prefix fuzz property) and P2C."""
import random

from infermesh.routing import (
    KVAwareRouter,
    PowerOfTwoChoicesRouter,
    RadixTrie,
    _cache_aware_pick,
)
from infermesh.settings import GatewayConfig, WorkerConfig


def test_trie_basic_match_and_remove():
    t = RadixTrie(ttl_s=1e9)
    t.insert("m", "hello world", "w1")
    wid, depth = t.longest_prefix_match("m", "hello world and more")
    assert wid == ["w1"] and depth == len("hello world")
    assert t.longest_prefix_match("m", "goodbye") == ([], 0)
    assert t.remove("m", "hello world", "w1") is True
    assert t.longest_prefix_match("m", "hello world")[0] == []


def test_trie_per_model_isolation():
    t = RadixTrie(ttl_s=1e9)
    t.insert("a", "shared", "wa")
    t.insert("b", "shared", "wb")
    assert t.longest_prefix_match("a", "shared")[0] == ["wa"]
    assert t.longest_prefix_match("b", "shared")[0] == ["wb"]


def test_shared_prefix_keeps_owner_after_branching():
    # Regression: a split node used to carry no worker, so a third request
    # with the same system prompt missed once two conversations had branched.
    sys_prompt = "system: You are the Acme support bot. " * 4 + "\nuser: "
    t = RadixTrie(ttl_s=1e9)
    t.insert("m", sys_prompt + "Can I return headphones?", "w1")
    t.insert("m", sys_prompt + "Can I exchange a laptop?", "w1")
    wid, depth = t.longest_prefix_match("m", sys_prompt + "Is software refundable?")
    assert wid == ["w1"] and depth >= len(sys_prompt)


def test_shared_prefix_tracks_every_worker_that_sent_it():
    t = RadixTrie(ttl_s=1e9)
    t.insert("m", "sys: shared|q1", "w1")
    t.insert("m", "sys: shared|q2", "w2")
    wid, depth = t.longest_prefix_match("m", "sys: shared|q3")
    assert sorted(wid) == ["w1", "w2"] and depth == len("sys: shared|q")
    assert t.longest_prefix_match("m", "sys: shared|q3", candidates={"w2"})[0] == ["w2"]


def test_remove_drops_worker_from_subtree_only():
    t = RadixTrie(ttl_s=1e9)
    t.insert("m", "abc-long", "w1")
    t.insert("m", "abc-long", "w2")
    assert t.remove("m", "abc", "w1") is True
    assert t.longest_prefix_match("m", "abc-long")[0] == ["w2"]


def test_longest_prefix_fuzz_never_below_naive():
    random.seed(1234)
    t = RadixTrie(ttl_s=1e9)
    keys = []
    for _ in range(600):
        k = "".join(random.choice("abc") for _ in range(random.randint(1, 8)))
        t.insert("m", k, "w")
        keys.append(k)

    def naive_longest_prefix(query):
        best = 0
        for k in keys:
            if query.startswith(k) and len(k) > best:
                best = len(k)
        return best

    for _ in range(3000):
        q = "".join(random.choice("abc") for _ in range(random.randint(1, 10)))
        wid, depth = t.longest_prefix_match("m", q)
        nb = naive_longest_prefix(q)
        assert depth >= nb, f"trie depth {depth} < naive {nb} for {q!r}"
        if nb > 0:
            assert wid == ["w"]


def test_trie_sibling_edges_and_shared_stem():
    t = RadixTrie(ttl_s=1e9)
    t.insert("m", "abc", "w_c")
    t.insert("m", "abd", "w_d")
    assert t.longest_prefix_match("m", "abc") == (["w_c"], 3)
    assert t.longest_prefix_match("m", "abd") == (["w_d"], 3)
    assert t.longest_prefix_match("m", "abcZZZ") == (["w_c"], 3)
    assert t.longest_prefix_match("m", "abdZZZ") == (["w_d"], 3)
    # The shared stem "ab" is held by both workers.
    wid, depth = t.longest_prefix_match("m", "abz")
    assert sorted(wid) == ["w_c", "w_d"] and depth == 2

    t.insert("m", "abcdef", "w_deep")
    assert t.longest_prefix_match("m", "abcdef") == (["w_deep"], 6)
    assert t.longest_prefix_match("m", "abcdefghi") == (["w_deep"], 6)
    assert t.longest_prefix_match("m", "abcdeQ") == (["w_deep"], 5)


class _StubRegistry:
    def __init__(self, workers):
        self._w = workers

    def get_workers(self, model):
        return [w for w in self._w if w.model == model]


def _vllm(wid, util):
    return WorkerConfig(id=wid, type="vllm", url="http://x", model="m", cache_utilization=util)


def test_p2c_prefers_lower_cache_utilization():
    reg = _StubRegistry([_vllm("A", 0.05), _vllm("B", 0.90)])
    p2c = PowerOfTwoChoicesRouter(etif_weight=0.6, etif_scale=4096)
    picks = {p2c.pick(reg.get_workers("m")).id for _ in range(50)}
    assert picks == {"A"}


def _openai(wid, inflight=0):
    return WorkerConfig(id=wid, type="openai", url="http://x", model="m",
                        requests_in_flight=inflight)


def cfg(**kw):
    return GatewayConfig(_env_file=None, **kw)


def test_match_must_cover_cache_threshold_of_prompt():
    t = RadixTrie(ttl_s=1e9)
    t.insert("m", "user: hello", "w1")
    pool = [_openai("w1"), _openai("w2")]
    # "user: " (6 chars) of a 60-char prompt is a coincidence, not reuse.
    assert _cache_aware_pick(t, "m", "user: " + "x" * 54, pool, cfg()) is None
    assert _cache_aware_pick(t, "m", "user: hello there", pool, cfg()).id == "w1"
    assert _cache_aware_pick(t, "m", "user: hello there", pool, cfg(kv_match_min_chars=64)) is None


def test_load_imbalance_overrides_cache():
    t = RadixTrie(ttl_s=1e9)
    t.insert("m", "system: long shared prompt", "busy")
    pool = [_openai("busy", inflight=100), _openai("idle", inflight=0)]
    assert _cache_aware_pick(t, "m", "system: long shared prompt", pool, cfg()) is None
    pool = [_openai("busy", inflight=10), _openai("idle", inflight=0)]
    assert _cache_aware_pick(t, "m", "system: long shared prompt", pool, cfg()).id == "busy"


def test_least_loaded_worker_among_matches():
    t = RadixTrie(ttl_s=1e9)
    t.insert("m", "system: shared", "a")
    t.insert("m", "system: shared", "b")
    pool = [_openai("a", inflight=5), _openai("b", inflight=1)]
    assert _cache_aware_pick(t, "m", "system: shared", pool, cfg()).id == "b"


def test_router_skips_unavailable_owner():
    t = RadixTrie(ttl_s=1e9)
    t.insert("m", "system: shared prompt", "w1")
    reg = _StubRegistry([_openai("w1"), _openai("w2")])
    router = KVAwareRouter(reg, t, cfg(), is_available=lambda w: w.id != "w1")
    worker, hit = router.select("m", [], prefix="system: shared prompt")
    assert worker.id == "w2" and hit is False
