"""ApiExecutor request-kwarg passing: temperature rides along ONLY when set.

The eval harness pins temperature 0.0 for comparable scorecards; every existing caller
leaves it None and must produce byte-identical request kwargs to before the knob existed
(no ``temperature`` key at all — provider defaults stay in charge).
"""

from __future__ import annotations

import asyncio
from typing import Any

from forge.shared.ensemble.executor import ApiExecutor
from forge.shared.ensemble.models import Prompt


class _Recorder:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None


def _fake_openai_class(recorder: _Recorder) -> type:
    class _Completions:
        async def create(self, **kwargs: Any) -> Any:
            recorder.kwargs = kwargs
            message = type("Message", (), {"content": "ok"})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    class _Chat:
        completions = _Completions()

    class FakeAsyncOpenAI:
        def __init__(self, **_: Any) -> None:
            self.chat = _Chat()

        async def __aenter__(self) -> FakeAsyncOpenAI:
            return self

        async def __aexit__(self, *exc: Any) -> bool:
            return False

    return FakeAsyncOpenAI


def _fake_anthropic_class(recorder: _Recorder) -> type:
    class _Messages:
        async def create(self, **kwargs: Any) -> Any:
            recorder.kwargs = kwargs
            block = type("Block", (), {"type": "text", "text": "ok"})()
            return type("Response", (), {"content": [block]})()

    class FakeAsyncAnthropic:
        def __init__(self, **_: Any) -> None:
            self.messages = _Messages()

        async def __aenter__(self) -> FakeAsyncAnthropic:
            return self

        async def __aexit__(self, *exc: Any) -> bool:
            return False

    return FakeAsyncAnthropic


def _call(executor: ApiExecutor, prompt: Prompt) -> str:
    return asyncio.run(executor._call(prompt))


def test_openai_omits_temperature_by_default(monkeypatch) -> None:
    recorder = _Recorder()
    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", _fake_openai_class(recorder))
    executor = ApiExecutor(label="router:m", kind="openai", model="m", base_url="http://x")
    assert _call(executor, Prompt(user="hi")) == "ok"
    assert recorder.kwargs is not None
    assert "temperature" not in recorder.kwargs


def test_openai_passes_temperature_when_set(monkeypatch) -> None:
    recorder = _Recorder()
    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", _fake_openai_class(recorder))
    executor = ApiExecutor(label="router:m", kind="openai", model="m", base_url="http://x")
    _call(executor, Prompt(user="hi", temperature=0.0))
    assert recorder.kwargs is not None
    assert recorder.kwargs["temperature"] == 0.0


def test_anthropic_omits_temperature_by_default(monkeypatch) -> None:
    recorder = _Recorder()
    import anthropic

    monkeypatch.setattr(anthropic, "AsyncAnthropic", _fake_anthropic_class(recorder))
    executor = ApiExecutor(label="anthropic:m", kind="anthropic", model="m", api_key="k")
    assert _call(executor, Prompt(user="hi")) == "ok"
    assert recorder.kwargs is not None
    assert "temperature" not in recorder.kwargs


def test_anthropic_passes_temperature_when_set(monkeypatch) -> None:
    recorder = _Recorder()
    import anthropic

    monkeypatch.setattr(anthropic, "AsyncAnthropic", _fake_anthropic_class(recorder))
    executor = ApiExecutor(label="anthropic:m", kind="anthropic", model="m", api_key="k")
    _call(executor, Prompt(user="hi", temperature=0.7))
    assert recorder.kwargs is not None
    assert recorder.kwargs["temperature"] == 0.7
