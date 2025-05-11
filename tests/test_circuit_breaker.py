"""CircuitBreaker state machine + failure classification."""
from infermesh.proxy import CircuitBreaker, is_breaker_failure


def test_is_breaker_failure_only_server_and_overload():
    assert is_breaker_failure(500)
    assert is_breaker_failure(502)
    assert is_breaker_failure(503)
    assert is_breaker_failure(429)
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