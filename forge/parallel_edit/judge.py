"""Judge that compares candidate diffs and produces a structured verdict.

The judge runs through the shared ensemble harness's failover ``Pool``: an ordered set of
interchangeable LLM executors. The primary is whatever ``judge_backend`` selects; if it is
pulled, rate-limited, or times out, the Pool fails over to the configured router models so
the comparison still happens. This is parallel_edit's first consumption of the harness.
"""

from __future__ import annotations

import json
import re

from agents.parallel_edit.config import settings
from agents.parallel_edit.models import (
    DimensionScores,
    EditRun,
    FileComparison,
    JudgeVerdict,
)
from agents.parallel_edit.prompts import build_judge_system_prompt
from agents.shared.ensemble import ApiExecutor, Pool, Prompt


def _build_judge_pool() -> Pool:
    """Assemble the judge failover pool: configured primary, then router failover models.

    Deduplicates by label so a primary that coincides with a failover entry isn't tried twice.
    """
    executors: list[ApiExecutor] = []
    seen: set[str] = set()

    def add(executor: ApiExecutor) -> None:
        if executor.label not in seen:
            seen.add(executor.label)
            executors.append(executor)

    if settings.judge_backend == "anthropic":
        add(
            ApiExecutor(
                label=f"anthropic:{settings.judge_anthropic_model}",
                kind="anthropic",
                model=settings.judge_anthropic_model,
            )
        )
    else:
        add(
            ApiExecutor(
                label=f"router:{settings.judge_openai_model}",
                kind="openai",
                model=settings.judge_openai_model,
                base_url=settings.judge_openai_base_url,
                api_key=settings.judge_openai_api_key,
            )
        )

    for model in settings.judge_failover_models:
        add(
            ApiExecutor(
                label=f"router:{model}",
                kind="openai",
                model=model,
                base_url=settings.judge_openai_base_url,
                api_key=settings.judge_openai_api_key,
            )
        )

    return Pool(
        role="judge",
        executors=executors,
        max_attempts_per_executor=settings.judge_max_attempts_per_model,
    )


def _judge_max_tokens() -> int:
    if settings.judge_backend == "anthropic":
        return settings.judge_anthropic_max_tokens
    return settings.judge_openai_max_tokens


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


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _status_note(run: EditRun) -> str:
    """Honest flag so the judge weighs completeness fairly for cut-off candidates."""
    if run.status == "timeout":
        return " (NOTE: cut off by a timeout — its diff is what it had written when killed)"
    if run.status == "error":
        return " (NOTE: candidate exited with an error — its diff may be incomplete)"
    return ""


def _format_candidate(run: EditRun, *, limit: int) -> str:
    diff, truncated = _truncate(run.diff_text, limit)
    truncated_note = " (DIFF TRUNCATED to fit context)" if truncated else ""
    stat = run.diff_stat
    if not run.diff_text.strip():
        body = "(No changes — this candidate produced no diff against the base.)"
    else:
        body = f"```diff\n{diff}\n```"
    return (
        f"## Candidate {run.label} — model: {run.model}{_status_note(run)}\n"
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


def _legacy_verdict_to_best(verdict: str) -> str:
    """Map a legacy pairwise per-file verdict ("A better", "B only", ...) to a `best` label."""
    verdict = verdict.strip()
    if verdict == "equivalent":
        return "equivalent"
    # "A better" / "B better" / "A only" / "B only" -> the leading label token.
    head = verdict.split(" ", 1)[0]
    return head or "equivalent"


def _parse_verdict(data: dict, run_labels: list[str]) -> JudgeVerdict | None:
    """Best-effort parse. Returns None if the response is too malformed to use."""
    if not isinstance(data, dict):
        return None

    winner_raw = str(data.get("winner", "")).strip()
    # "both_flawed" is the legacy pairwise spelling — normalize it so older judge models
    # (or cached prompts) that still emit it don't get rejected.
    if winner_raw == "both_flawed":
        winner_raw = "all_flawed"
    if winner_raw not in set(run_labels) | {"tie", "all_flawed"}:
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
        best = entry.get("best")
        if best is None and "verdict" in entry:
            best = _legacy_verdict_to_best(str(entry.get("verdict", "")))
        try:
            per_file_notes.append(
                FileComparison(
                    file=str(entry.get("file", "")).strip(),
                    best=str(best or "equivalent").strip(),
                    note=str(entry.get("note", "")).strip(),
                )
            )
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
    """Compare candidate runs via the failover judge pool. Returns (verdict, judge_model, error).

    ``judge_model`` is the label of whichever pool member actually answered (after any
    failover), so the report records who judged, not just who was asked first. Verdict is
    None when the judge could not produce a usable comparison (< 2 candidates with diffs,
    the whole pool exhausted, or an unparseable response).
    """
    pool = _build_judge_pool()
    primary_label = pool.executors[0].label if pool.executors else "judge"

    # A candidate is comparable if it produced a diff (even a cut-off timeout) or cleanly
    # made no changes — not only if it exited successfully. Discarding a thorough-but-slow
    # candidate's on-disk work is exactly the flaw the first live run surfaced.
    eligible = [r for r in runs if r.diff_text.strip() or r.status == "no_changes"]
    if len(eligible) < 2:
        return (
            None,
            primary_label,
            f"need at least 2 candidates with comparable output, got {len(eligible)}",
        )

    labels = [r.label for r in eligible]

    user_message = _build_user_message(prompt, eligible)
    judge_prompt = Prompt(
        system=build_judge_system_prompt(labels),
        user=user_message,
        max_tokens=_judge_max_tokens(),
    )

    def _produces_verdict(text: str) -> bool:
        return _parse_verdict(_extract_json(text), labels) is not None

    # The pool re-rolls and fails over until the output actually parses into a verdict, so a
    # model's JSON flakiness no longer aborts the comparison.
    result = await pool.run(
        judge_prompt, timeout=settings.judge_timeout_seconds, validate=_produces_verdict
    )
    judge_model = result.executor
    if not result.ok:
        err = f"no parseable verdict after {result.attempts} attempts ({result.error})"
        return None, judge_model, err

    verdict = _parse_verdict(_extract_json(result.output), labels)
    if verdict is None:  # validator passed, so this is unreachable in practice — defensive only
        return None, judge_model, "judge response did not match expected schema"

    return verdict, judge_model, None
