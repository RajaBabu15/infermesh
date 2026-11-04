"""E1 mock upstream: the metastable loop is reproduced and the budget breaks it."""
from bench.retry.sim import Params, Policy, run


def test_fixed_retries_sustain_a_metastable_failure_that_the_budget_prevents():
    p = Params()                                    # Bronson Sec 2.1: 300 rps capacity, 280 rps load, 10 s outage
    fixed = run(p, Policy("fixed2", "fixed"), seed=1)
    static = run(p, Policy("static10", "static"), seed=1)
    none = run(p, Policy("none", "none"), seed=1)
    assert fixed["recovered"] == 0.0 and fixed["steady_goodput_rps"] < 0.1 * p.load_rps
    assert static["recovered"] == 1.0 and static["steady_goodput_rps"] > 0.9 * p.load_rps
    assert none["recovered"] == 1.0
    assert fixed["attempts_per_request"] > static["attempts_per_request"]       # work amplification


def test_budget_never_sends_more_attempts_than_requests_plus_ten_percent_and_burst():
    p = Params(duration_s=60.0)
    r = run(p, Policy("static10", "static"), seed=2)
    assert r["attempts_per_request"] <= 1.0 + 0.10 + 13 / (p.load_rps * p.duration_s)


def test_deterministic_for_a_seed():
    p = Params(duration_s=40.0)
    assert run(p, Policy("headroom", "headroom"), seed=7) == run(p, Policy("headroom", "headroom"), seed=7)


def test_no_outage_means_no_retries_needed_and_full_goodput():
    p = Params(outage_s=0.0, duration_s=60.0)
    for policy in (Policy("fixed2", "fixed"), Policy("static10", "static")):
        r = run(p, policy, seed=3)
        assert r["recovered"] == 1.0 and r["success_ratio"] > 0.99


def _part_b(**kw):
    return Params(servers=4, load_rps=150.0, duration_s=60.0, warm_saving=0.8, prefixes=8,
                  outage_servers=(0,), **kw)


def test_outage_on_one_server_leaves_the_others_alive():
    r = run(_part_b(), Policy("none", "none"), seed=1)
    assert 0.70 < r["window_success_ratio"] < 0.80      # prefixes 0 and 4 of 8 live on the dead server


def test_cache_cost_admission_lets_a_static_budget_use_warm_fallbacks():
    # Retries fail over to servers 1..3, which hold the prefix warm after the first retry
    # and serve it at 20 percent of the cold cost. alpha = 0 prices every retry at one
    # credit, so a 10 percent budget starves them; alpha = 0.8 prices a warm retry at 0.2.
    flat = run(_part_b(), Policy("static", "static", alpha=0.0), seed=1)
    cost_aware = run(_part_b(), Policy("static", "static", alpha=0.8), seed=1)
    assert cost_aware["window_success_ratio"] > flat["window_success_ratio"] + 0.10
    assert cost_aware["window_success_ratio"] > 0.99
