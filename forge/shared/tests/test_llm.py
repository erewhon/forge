"""extract_json: plain JSON first, fences second, brace-scan last.

Regression suite for the fence-first bug the eval harness caught: valid JSON
whose STRING VALUES contain ``` sequences (markdown payload content) was
mangled by fence stripping and discarded — production replans included.
"""

from __future__ import annotations

import json

from agents.shared.llm import extract_json


def test_plain_json_passes_through():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced_json_is_extracted():
    text = 'Here is the answer:\n```json\n{"a": 1}\n```\nHope that helps!'
    assert extract_json(text) == {"a": 1}


def test_json_with_prose_around_it_brace_scan():
    text = 'Sure! {"a": 1} is what you want.'
    assert extract_json(text) == {"a": 1}


def test_valid_json_with_embedded_fences_survives():
    """The regression: ``` inside a JSON string value must not trigger fence
    stripping (an eval replan payload with a fenced example in its content was
    destroyed by the old fence-first order)."""
    payload = {
        "actions": [
            {
                "kind": "respec",
                "leaf_title": "x",
                "revised": {
                    "title": "x",
                    "content": "Run it:\n```bash\nunitconv list-units\n```\nthen check output.",
                    "feature": "F",
                },
            }
        ]
    }
    text = json.dumps(payload)
    assert "```" in text
    assert extract_json(text) == payload


def test_garbage_returns_empty():
    assert extract_json("not json at all") == {}


def test_truncated_json_returns_empty():
    assert extract_json('{"actions": [{"kind": "halt", "reason": "x"}]') == {}
