"""Single-LLM judge that compares two candidate diffs and produces a structured verdict."""

from __future__ import annotations

import json
import re

import anthropic
import openai

from agents.parallel_edit.config import settings
from agents.parallel_edit.models import (
    DimensionScores,
    EditRun,
    FileComparison,
    JudgeVerdict,
)
from agents.parallel_edit.prompts import JUDGE_SYSTEM_PROMPT


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of the LLM response, tolerating ```json fences."""
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

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


async def _complete_anthropic(system: str, user: str) -> str:
    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model=settings.judge_anthropic_model,
        max_tokens=settings.judge_anthropic_max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


async def _complete_openai(system: str, user: str) -> str:
    client = openai.AsyncOpenAI(
        base_url=settings.judge_openai_base_url, api_key=settings.judge_openai_api_key
    )
    response = await client.chat.completions.create(
        model=settings.judge_openai_model,
        max_tokens=settings.judge_openai_max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""


def _judge_model_name() -> str:
    if settings.judge_backend == "anthropic":
        return settings.judge_anthropic_model
    return settings.judge_openai_model


async def _complete(system: str, user: str) -> str:
    if settings.judge_backend == "anthropic":
        return await _complete_anthropic(system, user)
    return await _complete_openai(system, user)


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _format_candidate(run: EditRun, *, limit: int) -> str:
    diff, truncated = _truncate(run.diff_text, limit)
    truncated_note = " (DIFF TRUNCATED to fit context)" if truncated else ""
    stat = run.diff_stat
    if run.status == "no_changes":
        body = "(No changes — this candidate produced no diff against the base.)"
    else:
        body = f"```diff\n{diff}\n```"
    return (
        f"## Candidate {run.label} — model: {run.model}\n"
        f"Diff stat: {stat.files_changed} files, "
        f"+{stat.insertions} / -{stat.deletions}{truncated_note}\n\n"
        f"{body}"
    )


def _build_user_message(prompt: str, runs: list[EditRun]) -> str:
    parts = [
        "## Original change request",
        "",
        prompt,
        "",
    ]
    for run in runs:
        parts.append(_format_candidate(run, limit=settings.max_diff_chars_per_candidate))
        parts.append("")
    parts.append("Produce the comparison verdict as JSON now.")
    return "\n".join(parts)


def _parse_verdict(data: dict, run_labels: list[str]) -> JudgeVerdict | None:
    """Best-effort parse. Returns None if the response is too malformed to use."""
    if not isinstance(data, dict):
        return None

    winner_raw = str(data.get("winner", "")).strip()
    if winner_raw not in ("A", "B", "tie", "both_flawed"):
        return None

    scores_raw = data.get("scores") or {}
    scores: dict[str, DimensionScores] = {}
    for label in run_labels:
        candidate_scores = scores_raw.get(label)
        if not isinstance(candidate_scores, dict):
            continue
        try:
            scores[label] = DimensionScores(**candidate_scores)
        except Exception:
            continue
    if not scores:
        return None

    per_file_notes: list[FileComparison] = []
    for entry in data.get("per_file_notes", []) or []:
        if not isinstance(entry, dict):
            continue
        try:
            per_file_notes.append(FileComparison(**entry))
        except Exception:
            continue

    return JudgeVerdict(
        winner=winner_raw,  # type: ignore[arg-type]
        scores=scores,
        per_file_notes=per_file_notes,
        summary=str(data.get("summary", "")).strip(),
        recommendation=str(data.get("recommendation", "")).strip(),
    )


async def judge_runs(
    *, prompt: str, runs: list[EditRun]
) -> tuple[JudgeVerdict | None, str, str | None]:
    """Compare two candidate runs. Returns (verdict, judge_model, error_message).

    Verdict is None when the judge could not produce a usable comparison
    (e.g. < 2 candidates with diffs, LLM error, or unparseable response).
    """
    judge_model = _judge_model_name()
    eligible = [r for r in runs if r.status in ("ok", "no_changes")]
    if len(eligible) < 2:
        return (
            None,
            judge_model,
            f"need at least 2 successful candidates, got {len(eligible)}",
        )

    user_message = _build_user_message(prompt, eligible)

    try:
        response_text = await _complete(JUDGE_SYSTEM_PROMPT, user_message)
    except Exception as e:
        return None, judge_model, f"judge call failed: {type(e).__name__}: {e}"

    data = _extract_json(response_text)
    if not data:
        return None, judge_model, "judge response was not valid JSON"

    labels = [r.label for r in eligible]
    verdict = _parse_verdict(data, labels)
    if verdict is None:
        return None, judge_model, "judge response did not match expected schema"

    return verdict, judge_model, None
