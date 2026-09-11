"""ApiExecutor request construction: temperature rides along ONLY when set, and every router
call carries X-Router-Privacy.

The eval harness pins temperature 0.0 for comparable scorecards; every existing caller
leaves it None and must produce byte-identical request kwargs to before the knob existed
(no ``temperature`` key at all — provider defaults stay in charge).

Privacy: the tier is a required constructor argument (an absent header means "any" to the
router, so omission must be impossible), goes out as a default header on the OpenAI client,
comes back in ``ExecResult.meta`` alongside the router's routing headers, and a router privacy
refusal (403, ``privacy_tier_unavailable``) is reported as exactly that — terminal, named.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from forge.shared.ensemble.executor import ApiExecutor
from forge.shared.ensemble.models import ExecStatus, FailureClass, Prompt


class _Recorder:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None
        self.client_kwargs: dict[str, Any] | None = None


class _Raw:
    """Stands in for the SDK's LegacyAPIResponse: headers + parse()."""

    def __init__(self, parsed: Any, headers: dict[str, str]) -> None:
        self._parsed = parsed
        self.headers = headers

    def parse(self) -> Any:
        return self._parsed


def _fake_openai_class(
    recorder: _Recorder,
    *,
    response_headers: dict[str, str] | None = None,
    raise_exc: BaseException | None = None,
) -> type:
    class _RawCompletions:
        async def create(self, **kwargs: Any) -> Any:
            recorder.kwargs = kwargs
            if raise_exc is not None:
                raise raise_exc
            message = type("Message", (), {"content": "ok"})()
            choice = type("Choice", (), {"message": message})()
            return _Raw(type("Response", (), {"choices": [choice]})(), response_headers or {})

    class _Completions:
        with_raw_response = _RawCompletions()

    class _Chat:
        completions = _Completions()

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            recorder.client_kwargs = kwargs
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
    text, _meta = asyncio.run(executor._call(prompt))
    return text


def _router(**kw: Any) -> ApiExecutor:
    return ApiExecutor(
        label="router:m", kind="openai", model="m", privacy="zdr", base_url="http://x", **kw
    )


def test_openai_omits_temperature_by_default(monkeypatch) -> None:
    recorder = _Recorder()
    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", _fake_openai_class(recorder))
    assert _call(_router(), Prompt(user="hi")) == "ok"
    assert recorder.kwargs is not None
    assert "temperature" not in recorder.kwargs


def test_openai_passes_temperature_when_set(monkeypatch) -> None:
    recorder = _Recorder()
    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", _fake_openai_class(recorder))
    _call(_router(), Prompt(user="hi", temperature=0.0))
    assert recorder.kwargs is not None
    assert recorder.kwargs["temperature"] == 0.0


def test_anthropic_omits_temperature_by_default(monkeypatch) -> None:
    recorder = _Recorder()
    import anthropic

    monkeypatch.setattr(anthropic, "AsyncAnthropic", _fake_anthropic_class(recorder))
    executor = ApiExecutor(
        label="anthropic:m", kind="anthropic", model="m", privacy="any", api_key="k"
    )
    assert _call(executor, Prompt(user="hi")) == "ok"
    assert recorder.kwargs is not None
    assert "temperature" not in recorder.kwargs


def test_anthropic_passes_temperature_when_set(monkeypatch) -> None:
    recorder = _Recorder()
    import anthropic

    monkeypatch.setattr(anthropic, "AsyncAnthropic", _fake_anthropic_class(recorder))
    executor = ApiExecutor(
        label="anthropic:m", kind="anthropic", model="m", privacy="any", api_key="k"
    )
    _call(executor, Prompt(user="hi", temperature=0.7))
    assert recorder.kwargs is not None
    assert recorder.kwargs["temperature"] == 0.7


# --- X-Router-Privacy ---


def test_privacy_is_required() -> None:
    with pytest.raises(TypeError):
        ApiExecutor(label="router:m", kind="openai", model="m", base_url="http://x")  # type: ignore[call-arg]


def test_unknown_privacy_tier_is_refused_at_construction() -> None:
    # The router 403s an unrecognised value on every seat; catch the typo before the run.
    with pytest.raises(ValueError, match="unknown privacy tier"):
        ApiExecutor(label="router:m", kind="openai", model="m", privacy="locl", base_url="http://x")  # type: ignore[arg-type]


@pytest.mark.parametrize("tier", ["local", "zdr"])
def test_native_anthropic_cannot_honour_a_strict_tier(tier: str) -> None:
    """The native SDK bypasses the router; only 'any' is honest for it."""
    with pytest.raises(ValueError, match="bypasses the router"):
        ApiExecutor(label="anthropic:m", kind="anthropic", model="m", privacy=tier)  # type: ignore[arg-type]


def test_native_anthropic_accepts_any() -> None:
    executor = ApiExecutor(label="anthropic:m", kind="anthropic", model="m", privacy="any")
    assert executor.privacy == "any"


def test_openai_sends_privacy_header(monkeypatch) -> None:
    recorder = _Recorder()
    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", _fake_openai_class(recorder))
    _call(_router(), Prompt(user="hi"))
    assert recorder.client_kwargs is not None
    assert recorder.client_kwargs["default_headers"] == {"X-Router-Privacy": "zdr"}


def test_result_meta_carries_tier_and_router_routing_headers(monkeypatch) -> None:
    recorder = _Recorder()
    import openai

    monkeypatch.setattr(
        openai,
        "AsyncOpenAI",
        _fake_openai_class(
            recorder,
            response_headers={
                "X-Router-Privacy": "zdr",
                "X-Router-Resolved": "or/glm-5.2",
                "X-Router-Role": "",
            },
        ),
    )
    result = asyncio.run(_router().run(Prompt(user="hi"), timeout=5))
    assert result.ok
    assert result.meta == {
        "privacy": "zdr",
        "router_privacy": "zdr",
        "router_resolved": "or/glm-5.2",
    }


def test_result_meta_without_router_headers_still_names_the_tier(monkeypatch) -> None:
    recorder = _Recorder()
    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", _fake_openai_class(recorder))
    result = asyncio.run(_router().run(Prompt(user="hi"), timeout=5))
    assert result.meta == {"privacy": "zdr"}


class _StatusError(Exception):
    """Duck-types openai.APIStatusError: status_code + body (the unwrapped error object)."""

    def __init__(self, status_code: int, body: Any, message: str = "err") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def test_privacy_refusal_is_terminal_and_named(monkeypatch) -> None:
    """The router's 403 for 'no candidate satisfies the tier' must never be retried and must
    say why the seat is absent — not 'PermissionDeniedError ... check the model/key'."""
    recorder = _Recorder()
    import openai

    body = {
        "message": 'router refused: "glm" has no candidate that satisfies privacy tier "local"',
        "type": "privacy_policy_violation",
        "code": "privacy_tier_unavailable",
        "privacy_tier": "local",
        "excluded_candidates": ["or/glm-5.2: not on fleet hardware (privacy: local)"],
    }
    monkeypatch.setattr(
        openai,
        "AsyncOpenAI",
        _fake_openai_class(recorder, raise_exc=_StatusError(403, body)),
    )
    executor = ApiExecutor(
        label="router:glm", kind="openai", model="glm", privacy="local", base_url="http://x"
    )
    result = asyncio.run(executor.run(Prompt(user="hi"), timeout=5))
    assert result.status == ExecStatus.ERROR
    assert result.failure_class == FailureClass.TERMINAL
    assert result.error is not None
    assert result.error.startswith("refused by privacy policy (X-Router-Privacy: local)")
    assert "or/glm-5.2: not on fleet hardware" in result.error
    assert "will fail over, not retry" in result.error
    assert result.meta == {"privacy": "local", "privacy_refused": True}


def test_plain_403_is_still_a_generic_terminal_error(monkeypatch) -> None:
    recorder = _Recorder()
    import openai

    monkeypatch.setattr(
        openai,
        "AsyncOpenAI",
        _fake_openai_class(
            recorder, raise_exc=_StatusError(403, {"message": "bad key"}, "bad key")
        ),
    )
    result = asyncio.run(_router().run(Prompt(user="hi"), timeout=5))
    assert result.failure_class == FailureClass.TERMINAL
    assert result.error is not None
    assert not result.error.startswith("refused by privacy policy")
