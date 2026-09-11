"""extract_json: plain JSON first, fences second, brace-scan last.

Regression suite for the fence-first bug the eval harness caught: valid JSON
whose STRING VALUES contain ``` sequences (markdown payload content) was
mangled by fence stripping and discarded — production replans included.
"""

from __future__ import annotations

import json

import pytest

from forge.shared.llm import LLMConfig, complete, complete_with_retry, extract_json


def _cfg() -> LLMConfig:
    return LLMConfig(backend="openai", privacy="zdr")


def test_complete_with_retry_returns_first_nonempty(monkeypatch):
    calls: list[str] = []

    def fake(cfg, *, system, user_message, model, max_tokens=4096):
        calls.append(model)
        return "answer"

    monkeypatch.setattr("forge.shared.llm.complete", fake)
    out = complete_with_retry(_cfg(), system="s", user_message="u", model="research")
    assert out == "answer"
    assert len(calls) == 1  # succeeded first try, no retry


def test_complete_with_retry_retries_empty_then_succeeds(monkeypatch):
    outs = iter(["   ", "recovered"])
    monkeypatch.setattr("forge.shared.llm.complete", lambda *a, **k: next(outs))
    out = complete_with_retry(_cfg(), system="s", user_message="u", model="research")
    assert out == "recovered"


def test_complete_with_retry_gives_up_after_retries(monkeypatch):
    n = {"c": 0}

    def fake(*a, **k):
        n["c"] += 1
        return ""

    monkeypatch.setattr("forge.shared.llm.complete", fake)
    out = complete_with_retry(_cfg(), system="s", user_message="u", model="research", retries=1)
    assert out == ""  # still empty — caller must treat as failure
    assert n["c"] == 2  # initial attempt + one retry


# --- X-Router-Privacy on the sync helper ---


def test_privacy_is_required() -> None:
    with pytest.raises(TypeError):
        LLMConfig(backend="openai")  # type: ignore[call-arg]


def test_anthropic_backend_cannot_honour_a_strict_tier() -> None:
    with pytest.raises(ValueError, match="cannot honour"):
        LLMConfig(backend="anthropic", privacy="zdr")
    assert LLMConfig(backend="anthropic", privacy="any").privacy == "any"


def test_complete_sends_privacy_header(monkeypatch) -> None:
    import openai

    seen: dict = {}

    class _Completions:
        def create(self, **kwargs):
            message = type("Message", (), {"content": "ok"})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    class _Chat:
        completions = _Completions()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.chat = _Chat()

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    out = complete(
        LLMConfig(backend="openai", privacy="local"), system="s", user_message="u", model="coder"
    )
    assert out == "ok"
    assert seen["default_headers"] == {"X-Router-Privacy": "local"}


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
