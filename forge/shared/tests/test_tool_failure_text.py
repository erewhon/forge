"""Tool-proxy failure diagnostics must not be stored as research findings.

When the proxy's web tools fail it now reports why instead of returning an empty 200 — a real
improvement, but the text arrives in `content`, so it is non-empty and sails past the
empty-content guard. Observed live: a book sprint stored

    "(max tool rounds reached) — 11 of 12 tool call(s) failed. First failure — fetch_url: ..."

as a finding with confidence=low, and it reached the verifier as a genuine (if weak) answer.
"""

from __future__ import annotations

import pytest

from forge.shared.llm import is_tool_failure_text

FAILURES = [
    "(max tool rounds reached) — 11 of 12 tool call(s) failed. "
    "First failure — fetch_url: Invalid arguments: empty url",
    "(max tool rounds reached) — 4 of 12 tool call(s) failed. "
    "First failure — fetch_url: Fetch failed: HTTP 403",
    "(Max Tool Rounds Reached) — 1 of 3 tool call(s) failed.",
]

REAL_ANSWERS = [
    '{"question": "q", "answer": "Rayleigh scattering explains it.", "sources": ["s"]}',
    "The court found that the defendants inflated asset values between 2011 and 2015.",
    # a real answer that merely *discusses* tools must not trip the check
    "The agency failed to disclose its use of automated tools in the 2024 procurement round.",
    "",
]


@pytest.mark.parametrize("text", FAILURES)
def test_detects_proxy_failure_diagnostics(text: str) -> None:
    assert is_tool_failure_text(text)


@pytest.mark.parametrize("text", REAL_ANSWERS)
def test_leaves_real_answers_alone(text: str) -> None:
    assert not is_tool_failure_text(text)
