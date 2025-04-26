"""RadixTrie correctness (incl. a longest-prefix fuzz property) and P2C."""
import random

from infermesh.routing import PowerOfTwoChoicesRouter, RadixTrie, _kv_aware_pick
from infermesh.settings import WorkerConfig


def test_trie_basic_match_and_remove():
    t = RadixTrie(ttl_s=1e9)
    t.insert("m", "hello world", "w1")
    wid, depth = t.longest_prefix_match("m", "hello world and more")
    assert wid == "w1" and depth == len("hello world")
    assert t.longest_prefix_match("m", "goodbye") == (None, 0)
    assert t.remove("m", "hello world") is True
    assert t.longest_prefix_match("m", "hello world")[0] is None


def test_trie_per_model_isolation():
    t = RadixTrie(ttl_s=1e9)
    t.insert("a", "shared", "wa")
    t.insert("b", "shared", "wb")
    assert t.longest_prefix_match("a", "shared")[0] == "wa"
    assert t.longest_prefix_match("b", "shared")[0] == "wb"


def test_longest_prefix_fuzz_never_below_naive():
    """The trie's match depth must never be shorter than the longest inserted
    key that is a genuine prefix of the query (catches a premature early-return
    in the partial-match branch). Small alphabet => heavy prefix collisions."""
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
            assert wid is not None


class _StubRegistry:
    def __init__(self, workers):
        self._w = workers

    def get_workers(self, model):
        return [w for w in self._w if w.model == model]


def _vllm(wid, util):
    return WorkerConfig(id=wid, type="vllm", url="http://x", model="m", cache_utilization=util)


def test_p2c_prefers_lower_cache_utilization():
    reg = _StubRegistry([_vllm("A", 0.05), _vllm("B", 0.90)])
    p2c = PowerOfTwoChoicesRouter(reg, etif_weight=0.6, etif_scale=4096)
    picks = {p2c.select("m").id for _ in range(50)}
    # equal ETIF, both workers compared -> deterministic lower-util winner
    assert picks == {"A"}


def test_kv_match_min_chars_threshold():
    t = RadixTrie(ttl_s=1e9)
    t.insert("m", "ab", "w1")
    pool = [WorkerConfig(id="w1", type="vllm", url="http://x", model="m")]
    # depth-2 match counts with default (min 0 -> effectively 1)
    assert _kv_aware_pick(t, "m", "abcdef", pool, 0) is not None
    # require >=4 chars: the 2-char match is rejected
    assert _kv_aware_pick(t, "m", "abcdef", pool, 4) is None
