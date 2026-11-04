"""Upstream cool-down hints from rate-limit response headers.

`Retry-After` is RFC 9110: digits or an HTTP-date. Vendor reset hints are not:
Go durations (`6m0s`, `x-ratelimit-reset-*`), RFC 3339 timestamps
(`anthropic-ratelimit-*-reset`) and epoch seconds or milliseconds. Parsers return
None for anything unreadable, never inf or nan, so a bad header cannot hold a
circuit breaker open forever.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Mapping

DEFAULT_CAP_S = 300.0

_DIGITS = re.compile(r"^\d+(?:\.\d+)?$")
_DURATION = re.compile(r"^(?:\d+(?:\.\d+)?(?:ms|h|m|s))+$")
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT_S = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}

# Epoch values above these are treated as absolute timestamps, below as deltas.
_EPOCH_S_MIN = 1_000_000_000          # 2001-09-09 in seconds
_EPOCH_MS_MIN = 1_000_000_000_000     # the same instant in milliseconds

_RESET_HEADERS = (
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
    "x-ratelimit-reset",
    "anthropic-ratelimit-requests-reset",
    "anthropic-ratelimit-tokens-reset",
    "anthropic-ratelimit-input-tokens-reset",
    "anthropic-ratelimit-output-tokens-reset",
)


def _finite_nonneg(value: float) -> float | None:
    return value if math.isfinite(value) and value >= 0 else None


def parse_retry_after(value: str | None, now: datetime | None = None) -> float | None:
    """Seconds from a Retry-After value: digits or an HTTP-date. None otherwise."""
    if not value:
        return None
    value = value.strip()
    if _DIGITS.match(value):
        return _finite_nonneg(float(value))
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return _finite_nonneg(max(0.0, (when - (now or datetime.now(timezone.utc))).total_seconds()))


def parse_reset(value: str | None, now: datetime | None = None) -> float | None:
    """Seconds until a rate-limit window resets, from any of the vendor formats."""
    if not value:
        return None
    value = value.strip()
    now = now or datetime.now(timezone.utc)
    if _DURATION.match(value):
        return _finite_nonneg(sum(float(n) * _UNIT_S[u] for n, u in _DURATION_PART.findall(value)))
    if _DIGITS.match(value):
        number = float(value)
        if number >= _EPOCH_MS_MIN:
            return _finite_nonneg(max(0.0, number / 1000.0 - now.timestamp()))
        if number >= _EPOCH_S_MIN:
            return _finite_nonneg(max(0.0, number - now.timestamp()))
        return _finite_nonneg(number)
    if "T" in value:
        try:
            when = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return _finite_nonneg(max(0.0, (when - now).total_seconds()))
    return None


def cooldown_from_headers(
    headers: Mapping[str, str], cap_s: float = DEFAULT_CAP_S, now: datetime | None = None
) -> float | None:
    """How long to keep a worker out of rotation after a 429.

    Retry-After wins when valid. Otherwise the largest reset hint is used: the
    gateway cannot tell which limit (requests or tokens) was exceeded, and
    retrying early only burns more quota. The result is capped at `cap_s`.
    """
    lowered = {k.lower(): v for k, v in headers.items()}
    seconds = parse_retry_after(lowered.get("retry-after"), now)
    if seconds is None:
        hints = [h for h in (parse_reset(lowered.get(name), now) for name in _RESET_HEADERS) if h is not None]
        seconds = max(hints) if hints else None
    if seconds is None:
        return None
    return min(seconds, cap_s)
