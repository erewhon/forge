"""Executors: one concrete (endpoint, model) that runs a Prompt and returns an ExecResult.

ApiExecutor covers OpenAI-compatible (the local router) and Anthropic backends — the unit
that pr_review_ensemble's providers and the judge/aggregator all reduce to. SubprocessExecutor
(`claude -p` / opencode / codex) and ContainerExecutor (gaol dx) land when parallel_edit and
(c) are refactored on; they implement the same Executor protocol.
"""

from __future__ import annotations

import asyncio
import time
from typing import Protocol, runtime_checkable

from agents.shared.ensemble.classify import classify
from agents.shared.ensemble.models import ExecResult, ExecStatus, FailureClass, Prompt


@runtime_checkable
class Executor(Protocol):
    label: str

    async def run(self, prompt: Prompt, *, timeout: float) -> ExecResult: ...


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


class ApiExecutor:
    """A chat-completion executor for one (endpoint, model). kind is 'openai' or 'anthropic'."""

    def __init__(
        self,
        *,
        label: str,
        kind: str,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.label = label
        self.kind = kind
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    async def run(self, prompt: Prompt, *, timeout: float) -> ExecResult:
        start = time.monotonic()
        try:
            text = await asyncio.wait_for(self._call(prompt), timeout=timeout)
            return ExecResult(
                executor=self.label,
                status=ExecStatus.OK,
                output=text.strip(),
                latency_ms=_elapsed_ms(start),
            )
        except TimeoutError:
            return ExecResult(
                executor=self.label,
                status=ExecStatus.TIMEOUT,
                latency_ms=_elapsed_ms(start),
                error=f"timed out after {timeout:.0f}s",
                failure_class=FailureClass.TRANSIENT,
            )
        except Exception as exc:  # noqa: BLE001 — classify decides retry vs. fail over
            return ExecResult(
                executor=self.label,
                status=ExecStatus.ERROR,
                latency_ms=_elapsed_ms(start),
                error=f"{type(exc).__name__}: {exc}",
                failure_class=classify(exc),
            )

    async def _call(self, prompt: Prompt) -> str:
        if self.kind == "anthropic":
            import anthropic

            client = (
                anthropic.AsyncAnthropic(api_key=self.api_key)
                if self.api_key
                else anthropic.AsyncAnthropic()
            )
            response = await client.messages.create(
                model=self.model,
                max_tokens=prompt.max_tokens,
                system=prompt.system,
                messages=[{"role": "user", "content": prompt.user}],
            )
            for block in response.content:
                if block.type == "text":
                    return block.text
            return ""

        import openai

        client = openai.AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        messages: list[dict[str, str]] = []
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.user})
        response = await client.chat.completions.create(
            model=self.model, max_tokens=prompt.max_tokens, messages=messages
        )
        return response.choices[0].message.content or ""
