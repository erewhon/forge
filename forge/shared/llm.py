"""Shared LLM client helper for agent harnesses.

Wraps the two backend choices (Anthropic native client, OpenAI-compatible
client pointing at the local LLM router) behind a single `complete()`
function, plus a JSON-extraction helper for parsing LLM responses that
sometimes wrap their output in markdown code fences.

When `backend="openai"`, the `model` argument is passed through as-is —
this is the alias used by the local LiteLLM router (e.g. `research`,
`coder`). When `backend="anthropic"`, `model` is ignored and
`cfg.anthropic_model` is used instead, since the Anthropic backend has a
single model per harness rather than per-call routing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

Backend = Literal["openai", "anthropic"]


@dataclass(frozen=True)
class LLMConfig:
    backend: Backend
    openai_base_url: str = "http://localhost:4010/v1"
    openai_api_key: str = "sk-local-router"
    anthropic_model: str = "claude-sonnet-4-6"


def complete(
    cfg: LLMConfig,
    *,
    system: str,
    user_message: str,
    model: str,
    max_tokens: int = 4096,
) -> str:
    if cfg.backend == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=cfg.anthropic_model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        for block in response.content:
            if block.type == "text":
                return block.text.strip()
        return ""

    import openai

    client = openai.OpenAI(base_url=cfg.openai_base_url, api_key=cfg.openai_api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def extract_json(text: str) -> dict:
    """Extract JSON from an LLM response, tolerating markdown code fences."""
    code_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if code_match:
        text = code_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass
    return {}
