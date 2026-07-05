## SOURCE (retrylib/retry.py)

```python
import time


class RetryExhausted(Exception):
    """Raised when every attempt failed."""


def retry_call(fn, *, retries: int = 3, delay: float = 0.1):
    """Call fn(); on exception, retry up to `retries` times with linear backoff.

    Returns fn's result on the first success. Raises RetryExhausted (from the
    last error) when all attempts fail. `retries` counts ATTEMPTS, so
    retries=0 means the function is never called.
    """
    last_error = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(delay * (attempt + 1))
    raise RetryExhausted("all attempts failed") from last_error
```

## EXISTING TESTS (tests/test_retry.py)

```python
import pytest

from retrylib.retry import retry_call


def test_success_first_try():
    calls = []

    def ok():
        calls.append(1)
        return "done"

    assert retry_call(ok) == "done"
    assert len(calls) == 1


def test_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ValueError("transient")
        return "recovered"

    assert retry_call(flaky, retries=5) == "recovered"
    assert len(attempts) == 3
```
