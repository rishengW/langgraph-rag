from __future__ import annotations

import pytest

from src.utils.retry import call_with_retry, is_retryable_connection_error


def test_retryable_connection_marker_detection():
    assert is_retryable_connection_error(TimeoutError("connection timeout"))
    assert not is_retryable_connection_error(ValueError("bad input"))


def test_call_with_retry_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("src.utils.retry.time.sleep", lambda delay: None)
    attempts = {"count": 0}

    def operation():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("temporary timeout")
        return "ok"

    assert call_with_retry(operation, max_retries=2, log_label="unit") == "ok"
    assert attempts["count"] == 2


def test_call_with_retry_does_not_retry_non_retryable():
    attempts = {"count": 0}

    def operation():
        attempts["count"] += 1
        raise ValueError("bad")

    with pytest.raises(ValueError):
        call_with_retry(operation, max_retries=3)
    assert attempts["count"] == 1

