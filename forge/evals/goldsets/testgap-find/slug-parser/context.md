## SOURCE (textlib/slug.py)

```python
import re

_NON_WORD = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_len: int = 40) -> str:
    """Lowercase, replace non-alphanumeric runs with '-', trim to max_len.

    The trim never leaves a trailing '-': after cutting to max_len the slug
    is right-stripped of dashes.
    """
    slug = _NON_WORD.sub("-", text.lower()).strip("-")
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("-")
    return slug
```

## EXISTING TESTS (tests/test_slug.py)

```python
from textlib.slug import slugify


def test_basic():
    assert slugify("Hello World") == "hello-world"


def test_punctuation_runs():
    assert slugify("a -- b!!c") == "a-b-c"


def test_long_input_trimmed():
    assert len(slugify("word " * 30)) <= 40
```
