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

# Marker prefix stored in a research finding's ``answer`` when the LLM call produced nothing
# usable (empty content or a hard error). Research harnesses use it to tell a genuinely empty
# sprint apart from a real low-confidence finding, so a dead sprint can be aborted instead of
# silently scored 1/10. Keep callers in sync when matching on it.
RESEARCH_FAILED_PREFIX = "Research failed:"

# Signatures of a tool-proxy diagnostic returned *as if it were content*. When the proxy's web tools
# fail it now reports why instead of returning an empty 200 (a genuine improvement), but the text
# lands in `content`, so the harness stored strings like
#   "(max tool rounds reached) — 11 of 12 tool call(s) failed. First failure — fetch_url: HTTP 403"
# as a finding: non-empty, so the empty-content guard let it through, and it reached the verifier as
# a real (if low-confidence) answer. These are failures and must be recorded as such.
_TOOL_FAILURE_SIGNATURES = (
    "max tool rounds reached",
    "tool call(s) failed",
)


def is_tool_failure_text(text: str) -> bool:
    """True if `text` is a tool-proxy failure diagnostic rather than a real answer."""
    low = text.strip().lower()
    return any(sig in low for sig in _TOOL_FAILURE_SIGNATURES)


@dataclass(frozen=True)
class LLMConfig:
    backend: Backend
    openai_base_url: str = "http://localhost:4000/v1"
    openai_api_key: str = ""
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


def complete_with_retry(
    cfg: LLMConfig,
    *,
    system: str,
    user_message: str,
    model: str,
    max_tokens: int = 4096,
    retries: int = 1,
) -> str:
    """``complete()``, retrying when the model returns empty/whitespace content.

    An empty completion is the classic symptom of a transient tool-proxy / web-egress hiccup
    (see the 2026-07-24 outage): the request succeeds with HTTP 200 but carries no content. A
    single retry recovers most of these. Returns the final attempt's text, which may STILL be
    empty — callers must treat an empty return as a failure, not a valid finding.
    """
    text = ""
    for attempt in range(retries + 1):
        text = complete(
            cfg, system=system, user_message=user_message, model=model, max_tokens=max_tokens
        )
        if text.strip():
            return text
        if attempt < retries:
            print(f"    empty response from {model!r}; retrying ({attempt + 1}/{retries})...")
    return text


def extract_json(text: str) -> dict:
    """Extract JSON from an LLM response, tolerating markdown code fences.

    Plain valid JSON is tried FIRST: a ``` sequence inside a JSON string value
    (markdown content in a payload field, e.g. a LeafSpec spec with a fenced
    example) must not trigger fence stripping — the old fence-first order
    mangled such responses and discarded perfectly valid payloads."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    code_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if code_match:
        try:
            return json.loads(code_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    return {}
