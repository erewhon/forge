"""Executors: one concrete (endpoint, model) that runs a Prompt and returns an ExecResult.

ApiExecutor covers OpenAI-compatible (the local router) and Anthropic backends — the unit
that pr_review_ensemble's providers and the judge/aggregator all reduce to. SubprocessExecutor
(`claude -p` / opencode / codex) and ContainerExecutor (gaol dx) land when parallel_edit and
(c) are refactored on; they implement the same Executor protocol.

Every ApiExecutor carries a **privacy tier** (see :mod:`forge.shared.privacy`) and sends it as
``X-Router-Privacy`` on every router call. The argument is required, not defaulted: an absent
header means ``any`` to the router, and the one way to guarantee forge never sends "anything
goes" by accident is to make every construction site say what it wants.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Protocol, runtime_checkable

from forge.shared.ensemble.classify import classify
from forge.shared.ensemble.models import ExecResult, ExecStatus, FailureClass, Prompt
from forge.shared.privacy import (
    PrivacyTier,
    check_tier,
    privacy_headers,
    privacy_refusal,
    router_meta,
)
from forge.shared.usage import record_usage


@runtime_checkable
class Executor(Protocol):
    label: str

    async def run(self, prompt: Prompt, *, timeout: float) -> ExecResult: ...


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


class ApiExecutor:
    """A chat-completion executor for one (endpoint, model). kind is 'openai' or 'anthropic'.

    ``privacy`` is the X-Router-Privacy tier for the router (``kind="openai"``). The native
    Anthropic SDK (``kind="anthropic"``) bypasses the router entirely, so it can only honour
    ``"any"``: asking it for ``local`` or ``zdr`` is a configuration error and is refused at
    construction rather than silently served from Anthropic's retaining API.
    """

    def __init__(
        self,
        *,
        label: str,
        kind: str,
        model: str,
        privacy: PrivacyTier,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.label = label
        self.kind = kind
        self.model = model
        self.privacy = check_tier(privacy)
        self.base_url = base_url
        self.api_key = api_key
        if kind == "anthropic" and self.privacy != "any":
            raise ValueError(
                f"{label}: the native Anthropic backend bypasses the router and cannot honour "
                f"X-Router-Privacy: {self.privacy}; route the model through the router "
                "(kind='openai' with the router base_url) or set privacy='any' explicitly"
            )

    async def run(self, prompt: Prompt, *, timeout: float) -> ExecResult:
        start = time.monotonic()
        try:
            text, meta = await asyncio.wait_for(self._call(prompt), timeout=timeout)
            return ExecResult(
                executor=self.label,
                status=ExecStatus.OK,
                output=text.strip(),
                latency_ms=_elapsed_ms(start),
                meta=meta,
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
            refusal = privacy_refusal(exc)
            if refusal is not None:
                # The router's 403 for "no candidate satisfies the tier". Terminal by nature —
                # the answer is the same until the config or the tier changes — and named as
                # such, so a panel report says WHY the seat is absent instead of "403".
                return ExecResult(
                    executor=self.label,
                    status=ExecStatus.ERROR,
                    latency_ms=_elapsed_ms(start),
                    error=f"{refusal} [terminal — will fail over, not retry]",
                    failure_class=FailureClass.TERMINAL,
                    meta={"privacy": self.privacy, "privacy_refused": True},
                )
            fclass = classify(exc)
            # Name the retry disposition in the string too — a reader of the error alone (logs,
            # aggregated advisories) shouldn't have to cross-reference failure_class to know
            # whether a retry can help.
            disposition = {
                FailureClass.TERMINAL: "terminal — will fail over, not retry (check the model/key)",
                FailureClass.TRANSIENT: "transient — will retry, then fail over",
            }.get(fclass, "")
            error = f"{type(exc).__name__}: {exc}"
            return ExecResult(
                executor=self.label,
                status=ExecStatus.ERROR,
                latency_ms=_elapsed_ms(start),
                error=f"{error} [{disposition}]" if disposition else error,
                failure_class=fclass,
            )

    async def _call(self, prompt: Prompt) -> tuple[str, dict[str, Any]]:
        """The raw call: returns (text, meta). ``meta`` always carries the tier sent, plus the
        router's routing/privacy response headers when it set any (role/chain-resolved
        requests), so a run can record which concrete model served the seat under which tier."""
        # Each call opens its own client inside an ``async with`` so the underlying httpx
        # transport is closed *within* this event loop. Pools drive several sequential
        # ``asyncio.run`` loops, and a client left open gets GC'd against an already-closed loop —
        # the harmless-but-noisy "RuntimeError: Event loop is closed" at interpreter shutdown.
        # Closing it here silences that.
        sampling = {} if prompt.temperature is None else {"temperature": prompt.temperature}
        meta: dict[str, Any] = {"privacy": self.privacy}
        if self.kind == "anthropic":
            import anthropic

            kwargs = {"api_key": self.api_key} if self.api_key else {}
            async with anthropic.AsyncAnthropic(**kwargs) as client:
                response = await client.messages.create(
                    model=self.model,
                    max_tokens=prompt.max_tokens,
                    system=prompt.system,
                    messages=[{"role": "user", "content": prompt.user}],
                    **sampling,
                )
            usage = getattr(response, "usage", None)
            if usage is not None:
                record_usage(getattr(usage, "input_tokens", 0), getattr(usage, "output_tokens", 0))
            for block in response.content:
                if block.type == "text":
                    return block.text, meta
            return "", meta

        import openai

        messages: list[dict[str, str]] = []
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.user})
        async with openai.AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            default_headers=privacy_headers(self.privacy),
        ) as client:
            # with_raw_response: the router reports what it enforced and who served the request
            # in response headers (X-Router-Privacy / -Resolved / -Overflow); parse() yields the
            # same completion object the plain call would.
            raw = await client.chat.completions.with_raw_response.create(
                model=self.model, max_tokens=prompt.max_tokens, messages=messages, **sampling
            )
        meta.update(router_meta(getattr(raw, "headers", None)))
        response = raw.parse()
        usage = getattr(response, "usage", None)
        if usage is not None:
            record_usage(getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0))
        return response.choices[0].message.content or "", meta
