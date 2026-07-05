## SOURCE (cachelib/bounded.py)

```python
class BoundedCache:
    """A size-bounded mapping that evicts the OLDEST INSERTED key when full.

    Not thread-safe by design? Nothing documents it either way — callers in
    the async pipeline share one instance across workers.
    """

    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self._data: dict = {}

    def put(self, key, value) -> None:
        if key not in self._data and len(self._data) >= self.capacity:
            oldest = next(iter(self._data))
            del self._data[oldest]
        self._data[key] = value

    def get(self, key, default=None):
        return self._data.get(key, default)

    def prune(self, predicate) -> int:
        """Remove entries whose value matches predicate; returns removed count."""
        removed = 0
        for key in self._data:
            if predicate(self._data[key]):
                del self._data[key]
                removed += 1
        return removed
```

## EXISTING TESTS (tests/test_bounded.py)

```python
from cachelib.bounded import BoundedCache


def test_put_get_roundtrip():
    c = BoundedCache(capacity=2)
    c.put("a", 1)
    assert c.get("a") == 1


def test_missing_key_default():
    c = BoundedCache(capacity=2)
    assert c.get("nope", 42) == 42
```
