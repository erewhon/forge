from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

import anthropic
import openai

from agents.pr_review_ensemble.config import settings
from agents.pr_review_ensemble.models import ProviderName, ProviderReview


async def _measure(
    coro_factory: Callable[[], Awaitable[str]],
    *,
    provider: ProviderName,
    model: str,
    timeout: float,
) -> ProviderReview:
    start = time.monotonic()
    try:
        text = await asyncio.wait_for(coro_factory(), timeout=timeout)
        return ProviderReview(
            provider=provider,
            model=model,
            status="ok",
            response_text=text.strip(),
            latency_ms=int((time.monotonic() - start) * 1000),
        )
    except TimeoutError:
        return ProviderReview(
            provider=provider,
            model=model,
            status="timeout",
            latency_ms=int((time.monotonic() - start) * 1000),
            error_message=f"Timed out after {timeout:.1f}s",
        )
    except Exception as e:
        return ProviderReview(
            provider=provider,
            model=model,
            status="error",
            latency_ms=int((time.monotonic() - start) * 1000),
            error_message=f"{type(e).__name__}: {e}",
        )


async def _call_anthropic(system_prompt: str, user_message: str) -> ProviderReview:
    if not settings.anthropic_enabled:
        return ProviderReview(
            provider="anthropic",
            model=settings.anthropic_model,
            status="skipped",
            error_message="disabled in config",
        )

    client = anthropic.AsyncAnthropic()

    async def _call() -> str:
        response = await client.messages.create(
            model=settings.anthropic_model,
            max_tokens=settings.anthropic_max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""

    return await _measure(
        _call,
        provider="anthropic",
        model=settings.anthropic_model,
        timeout=settings.per_provider_timeout_seconds,
    )


async def _call_local(system_prompt: str, user_message: str) -> ProviderReview:
    if not settings.local_enabled:
        return ProviderReview(
            provider="local",
            model=settings.local_model,
            status="skipped",
            error_message="disabled in config",
        )

    client = openai.AsyncOpenAI(
        base_url=settings.local_base_url, api_key=settings.local_api_key
    )

    async def _call() -> str:
        response = await client.chat.completions.create(
            model=settings.local_model,
            max_tokens=settings.local_max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content or ""

    return await _measure(
        _call,
        provider="local",
        model=settings.local_model,
        timeout=settings.per_provider_timeout_seconds,
    )


async def _call_opencode_zen(system_prompt: str, user_message: str) -> ProviderReview:
    if not settings.opencode_zen_enabled:
        return ProviderReview(
            provider="opencode_zen",
            model=settings.opencode_zen_model,
            status="skipped",
            error_message="disabled in config",
        )
    if not settings.opencode_zen_api_key:
        return ProviderReview(
            provider="opencode_zen",
            model=settings.opencode_zen_model,
            status="skipped",
            error_message="PR_REVIEW_ENSEMBLE_OPENCODE_ZEN_API_KEY not set",
        )

    client = openai.AsyncOpenAI(
        base_url=settings.opencode_zen_base_url, api_key=settings.opencode_zen_api_key
    )

    async def _call() -> str:
        response = await client.chat.completions.create(
            model=settings.opencode_zen_model,
            max_tokens=settings.opencode_zen_max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content or ""

    return await _measure(
        _call,
        provider="opencode_zen",
        model=settings.opencode_zen_model,
        timeout=settings.per_provider_timeout_seconds,
    )


_DISPATCH: dict[ProviderName, Callable[[str, str], Awaitable[ProviderReview]]] = {
    "anthropic": _call_anthropic,
    "local": _call_local,
    "opencode_zen": _call_opencode_zen,
}


async def call_provider(
    provider: ProviderName, *, system_prompt: str, user_message: str
) -> ProviderReview:
    return await _DISPATCH[provider](system_prompt, user_message)


def all_providers() -> list[ProviderName]:
    return list(_DISPATCH.keys())
