"""CircuitBreaker state machine + failure classification."""
import time

import httpx

from infermesh.proxy import CircuitBreaker, is_breaker_failure, retry_after_s


def test_is_breaker_failure_only_server_errors():
    assert is_breaker_failure(500)
    assert is_breaker_failure(502)
    assert is_breaker_failure(503)
    assert not is_breaker_failure(429)  # cooldown via Retry-After instead
    assert not is_breaker_failure(400)
    assert not is_breaker_failure(401)
    assert not is_breaker_failure(403)
    assert not is_breaker_failure(404)
    assert not is_breaker_failure(422)
    assert not is_breaker_failure(200)


def test_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=3)
    cb.record_failure("w")
    cb.record_failure("w")
    assert cb.state == "CLOSED"
    cb.record_failure("w")
    assert cb.state == "OPEN"


def test_open_blocks_within_recovery_window():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout_s=999.0)
    cb.record_failure("w")
    assert cb.state == "OPEN"
    assert cb.allow_request() is False


def test_half_open_admits_single_probe():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout_s=0.0)
    cb.record_failure("w")
    assert cb.allow_request() is True
    assert cb.state == "HALF_OPEN"
    assert cb.allow_request() is False
    assert cb.allow_request() is False


def test_half_open_failure_reopens_immediately():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout_s=0.0)
    cb.record_failure("w")
    assert cb.allow_request() is True
    cb.record_failure("w")
    assert cb.state == "OPEN"
    assert cb.half_open_inflight is False


def test_half_open_success_closes_and_reopens_traffic():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout_s=0.0)
    cb.record_failure("w")
    assert cb.allow_request() is True
    cb.record_success()
    assert cb.state == "CLOSED"
    assert cb.failure_count == 0
    assert cb.allow_request() is True
    assert cb.allow_request() is True

def test_cool_down_honours_retry_after():
    cb = CircuitBreaker(failure_threshold=5, recovery_timeout_s=30.0)
    cb.cool_down(0.0, "w")
    assert cb.state == "OPEN"
    assert cb.allow_request() is True  # 0 s cooldown elapsed: probe allowed
    cb2 = CircuitBreaker(failure_threshold=5, recovery_timeout_s=30.0)
    cb2.cool_down(None, "w")  # no header: fall back to recovery timeout
    assert cb2.allow_request() is False


def test_retry_after_parsing():
    assert retry_after_s(httpx.Headers({"retry-after": "7"})) == 7.0
    assert retry_after_s(httpx.Headers({})) is None
    assert retry_after_s(httpx.Headers({"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"})) == 0.0
    assert retry_after_s(httpx.Headers({"retry-after": "soon"})) is None


def test_release_probe_unwedges_half_open():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout_s=0.0)
    cb.record_failure("w")
    assert cb.allow_request() is True      # probe admitted
    assert cb.allow_request() is False     # second caller blocked
    cb.release_probe()                     # probe ended with no verdict
    assert cb.allow_request() is True


def test_is_available_is_read_only():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout_s=999.0)
    assert cb.is_available()
    cb.record_failure("w")
    assert not cb.is_available()
    cb.retry_at = time.monotonic() - 1
    assert cb.is_available()
    assert cb.state == "OPEN"              # checking did not consume the probe


def _remaining(cb):
    return cb.retry_at - time.monotonic()


def test_ejection_time_is_base_times_consecutive_opens_and_is_capped():
    # Envoy: real time = base_ejection_time * ejection count, cap max_ejection_time.
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout_s=10.0, max_recovery_timeout_s=40.0)
    cb.record_failure("w")
    assert 9.5 < _remaining(cb) <= 10.0
    for expected in (20.0, 30.0, 40.0, 40.0):
        cb.retry_at = time.monotonic()           # recovery elapsed
        assert cb.allow_request() is True        # half-open probe
        cb.record_failure("w")                   # probe fails
        assert expected - 0.5 < _remaining(cb) <= expected
    cb.retry_at = time.monotonic()
    assert cb.allow_request() is True
    cb.record_success()
    assert cb.consecutive_opens == 0
    cb.record_failure("w")
    assert 9.5 < _remaining(cb) <= 10.0          # back to the base time


def test_explicit_cooldown_is_not_multiplied():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout_s=10.0)
    cb.record_failure("w")
    cb.cool_down(3.0, "w")
    assert 2.5 < _remaining(cb) <= 3.0
