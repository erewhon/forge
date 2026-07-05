## SOURCE (mathlib/clamp.py)

```python
def clamp(value: float, low: float, high: float) -> float:
    """Clamp value into [low, high]. Raises ValueError when low > high."""
    if low > high:
        raise ValueError(f"low {low} > high {high}")
    return max(low, min(value, high))
```

## EXISTING TESTS (tests/test_clamp.py)

```python
import pytest

from mathlib.clamp import clamp


@pytest.mark.parametrize(
    ("value", "low", "high", "expected"),
    [
        (5, 0, 10, 5),        # inside
        (-1, 0, 10, 0),       # below
        (11, 0, 10, 10),      # above
        (0, 0, 10, 0),        # at low boundary
        (10, 0, 10, 10),      # at high boundary
        (5, 5, 5, 5),         # degenerate interval
        (-0.0, 0.0, 1.0, 0.0),  # signed zero
    ],
)
def test_clamp_table(value, low, high, expected):
    assert clamp(value, low, high) == expected


def test_clamp_low_above_high_raises():
    with pytest.raises(ValueError, match="> high"):
        clamp(1, 10, 0)
```
