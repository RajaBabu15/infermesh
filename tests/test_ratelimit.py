"""Rate-limit header parsing: Retry-After and vendor reset hints."""
from datetime import datetime, timezone

import httpx
import pytest

from infermesh.proxy import retry_after_s
from infermesh.ratelimit import cooldown_from_headers, parse_reset, parse_retry_after

NOW = datetime(2026, 11, 3, 2, 0, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("value", ["inf", "-inf", "nan", "Infinity", "1e999", "-5", "soon", "", "  "])
def test_retry_after_rejects_non_finite_and_garbage(value):
    assert parse_retry_after(value, NOW) is None


def test_retry_after_inf_no_longer_returns_infinity():
    # Before the fix float("inf") passed through and kept the breaker open forever.
    assert retry_after_s(httpx.Headers({"retry-after": "inf"})) is None
    assert retry_after_s(httpx.Headers({"retry-after": "1e999"})) is None


def test_retry_after_accepts_digits_and_http_date():
    assert parse_retry_after("7", NOW) == 7.0
    assert parse_retry_after("2.5", NOW) == 2.5
    assert parse_retry_after("Tue, 03 Nov 2026 02:00:30 GMT", NOW) == 30.0
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT", NOW) == 0.0


@pytest.mark.parametrize("value,seconds", [
    ("42s", 42.0),
    ("6m0s", 360.0),
    ("1h30m", 5400.0),
    ("120ms", 0.12),
    ("1m30.5s", 90.5),
    ("2h", 7200.0),
])
def test_reset_duration_strings(value, seconds):
    assert parse_reset(value, NOW) == pytest.approx(seconds)


def test_reset_rfc3339_and_epoch():
    assert parse_reset("2026-11-03T02:15:00Z", NOW) == 900.0
    assert parse_reset("2026-11-03T02:15:00+00:00", NOW) == 900.0
    assert parse_reset("2026-11-03T01:00:00Z", NOW) == 0.0           # already past
    assert parse_reset(str(int(NOW.timestamp()) + 90), NOW) == 90.0  # epoch seconds
    assert parse_reset(str(int(NOW.timestamp() * 1000) + 90_000), NOW) == 90.0  # epoch ms
    assert parse_reset("15", NOW) == 15.0                            # small number is a delta


@pytest.mark.parametrize("value", ["inf", "nan", "tomorrow", "6x", "m5", ""])
def test_reset_rejects_garbage(value):
    assert parse_reset(value, NOW) is None


def test_cooldown_prefers_retry_after_then_largest_reset_and_caps():
    assert cooldown_from_headers({"Retry-After": "7", "x-ratelimit-reset-tokens": "6m0s"}, now=NOW) == 7.0
    assert cooldown_from_headers(
        {"x-ratelimit-reset-requests": "1s", "x-ratelimit-reset-tokens": "30s"}, now=NOW) == 30.0
    assert cooldown_from_headers({"anthropic-ratelimit-tokens-reset": "2026-11-03T02:01:00Z"}, now=NOW) == 60.0
    assert cooldown_from_headers({"x-ratelimit-reset-tokens": "6m0s"}, cap_s=300.0, now=NOW) == 300.0
    assert cooldown_from_headers({"Retry-After": "inf"}, now=NOW) is None
    assert cooldown_from_headers({}, now=NOW) is None
