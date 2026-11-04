"""RadixTrie correctness (incl. a longest-prefix fuzz property) and P2C."""
import random
import time

from conftest import cfg
from infermesh.routing import (
    KVAwareRouter,
    PowerOfTwoChoicesRouter,
    _cache_aware_pick,
)
from infermesh.settings import WorkerConfig
from infermesh.trie import RadixTrie


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


def _attributed_chars(trie, worker_id):
    """Recompute a worker's size from the nodes, independent of the running counter."""
    return sum(len(n.label) for root in trie._roots.values() for n in trie._walk(root)
               if worker_id in n.tenants)


def test_deep_trie_does_not_hit_the_recursion_limit():
    t = RadixTrie(ttl_s=1e9)
    for n in range(1, 3001):                  # "a", "aa", ... each a prefix of the next: depth 3000
        t.insert("m", "a" * n, "w1")
    assert t.entry_count() == 3000            # recursive version raised RecursionError here
    assert t.longest_prefix_match("m", "a" * 3000)[1] == 3000
    assert t.remove("m", "a" * 10, "w1") is True
    assert t.entry_count() == 9
    assert t.evict_expired() == 0
    assert t.remove_worker("w1") == 9
    assert t.entry_count() == 0


def test_expired_entries_are_pruned_iteratively():
    t = RadixTrie(ttl_s=1e9)
    for n in range(1, 2001):
        t.insert("m", "b" * n, "w1", ttl_override_s=0.0)
    time.sleep(0.01)
    assert t.evict_expired() == 2000
    assert t._roots == {}


def test_children_are_keyed_by_first_character():
    t = RadixTrie(ttl_s=1e9)
    for word in ("apple", "apricot", "banana", "band", "bandana", "cherry"):
        t.insert("m", word, "w1")
    for root in t._roots.values():
        for node in t._walk(root):
            assert all(k == child.label[0] and len(k) == 1 for k, child in node.children.items())
            assert all(child.parent is node for child in node.children.values())


def test_cap_keeps_a_worker_under_its_limit_and_drops_oldest_first():
    t = RadixTrie(ttl_s=1e9, max_chars_per_worker=200)
    prompts = [f"{i:03d}|" + "x" * 36 for i in range(20)]            # 40 chars each, all unique
    for p in prompts:
        t.insert("m", p, "w1")
        time.sleep(0.001)
    assert t.chars_for("w1") <= 200
    assert t.chars_for("w1") == _attributed_chars(t, "w1")
    # The oldest prompt's own tail is gone (only the few characters it shares with
    # newer prompts remain); the newest prompt is held in full.
    assert t.longest_prefix_match("m", prompts[0])[1] < len(prompts[0])
    assert t.longest_prefix_match("m", prompts[-1]) == (["w1"], len(prompts[-1]))


def test_a_recent_match_protects_an_old_prefix_from_eviction():
    t = RadixTrie(ttl_s=1e9, max_chars_per_worker=200)
    first = "000|" + "x" * 36
    t.insert("m", first, "w1")
    for i in range(1, 20):
        t.insert("m", f"{i:03d}|" + "x" * 36, "w1")     # keeps pushing the worker over its cap
        time.sleep(0.001)
        t.longest_prefix_match("m", first)               # but `first` stays recently used
    assert t.longest_prefix_match("m", first) == (["w1"], len(first))   # kept in full
    assert t.chars_for("w1") <= 200
    assert t.longest_prefix_match("m", "001|" + "x" * 36)[1] < 40       # an untouched one was evicted


def test_cap_is_per_worker():
    t = RadixTrie(ttl_s=1e9, max_chars_per_worker=100)
    for i in range(10):
        t.insert("m", f"a{i}|" + "x" * 30, "w1")
    t.insert("m", "keep|" + "y" * 30, "w2")
    assert t.chars_for("w1") <= 100
    assert t.longest_prefix_match("m", "keep|" + "y" * 30)[0] == ["w2"]


def test_char_accounting_matches_the_tree_after_random_operations():
    rng = random.Random(7)
    t = RadixTrie(ttl_s=1e9, max_chars_per_worker=400)
    alphabet = "ab"
    workers = ["w1", "w2", "w3"]
    for _ in range(600):
        prefix = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 30)))
        w = rng.choice(workers)
        op = rng.random()
        if op < 0.7:
            t.insert("m", prefix, w)
        elif op < 0.9:
            t.remove("m", prefix, w)
        else:
            t.longest_prefix_match("m", prefix)
    for w in workers:
        assert t.chars_for(w) == _attributed_chars(t, w)
        assert t.chars_for(w) <= 400
    for root in t._roots.values():            # no empty leaves left behind
        for node in t._walk(root):
            assert node.parent is None or node.tenants or node.children


def test_remove_worker_forgets_one_worker_only():
    t = RadixTrie(ttl_s=1e9)
    t.insert("m", "shared|one", "w1")
    t.insert("m", "shared|one", "w2")
    t.insert("other", "solo", "w1")
    assert t.remove_worker("w1") >= 2
    assert t.longest_prefix_match("m", "shared|one")[0] == ["w2"]
    assert t.longest_prefix_match("other", "solo")[0] == []
    assert t.chars_for("w1") == 0
    assert t.remove_worker("nobody") == 0


def test_common_len_matches_the_naive_loop():
    from infermesh.trie import _common_len

    def naive(a, b):
        i = 0
        while i < min(len(a), len(b)) and a[i] == b[i]:
            i += 1
        return i

    rng = random.Random(11)
    for _ in range(2000):
        base = "".join(rng.choice("ab") for _ in range(rng.randint(0, 40)))
        a = base + "".join(rng.choice("abc") for _ in range(rng.randint(0, 10)))
        b = base + "".join(rng.choice("abc") for _ in range(rng.randint(0, 10)))
        assert _common_len(a, b) == naive(a, b)
    assert _common_len("", "abc") == 0 and _common_len("abc", "") == 0
    long = "x" * 100_000
    assert _common_len(long, long + "y") == 100_000
    assert _common_len(long + "a", long + "b") == 100_000


def test_evicting_a_prefix_drops_it_and_its_extensions_but_keeps_shorter_ancestors():
    # kv_evicted says the engine dropped this prefix (and so everything after it).
    # Engines evict from the tail, so a shorter shared stem can still be cached;
    # it leaves the trie by its own eviction event or by TTL, not as a side effect.
    t = RadixTrie(ttl_s=1e9)
    t.insert("m", "apple", "w1")
    t.insert("m", "application", "w1")
    assert t.remove("m", "apple", "w1") is True
    assert t.longest_prefix_match("m", "apple") == (["w1"], 4)             # only the stem "appl"
    assert t.longest_prefix_match("m", "application") == (["w1"], 11)      # sibling branch untouched
    assert t.remove("m", "application", "w1") is True
    assert t.longest_prefix_match("m", "application") == (["w1"], 4)       # stem survives
    assert t.remove("m", "appl", "w1") is True                             # its own eviction removes it
    assert t.longest_prefix_match("m", "application") == ([], 0)
    assert t._roots == {} or not t._roots["m"].children
