"""Scorecard runner — model id → load gold sets → render prompts → call model → grade → aggregate.

CRITICAL DESIGN RULE: call ``Pool.run`` with NO ``validate`` callback.
``panel.structured`` retries schema-invalid output until the pool yields a
valid payload — for production that is resilience, for an eval it is grade
laundering. Transport retries stay; validity is graded from the first returned
text of each repeat.

The ``review-findings`` step runs the FULL production funnel per repeat —
finder call, slug-dedup + severity cap (mirroring ``collect_findings``), then
one confirm vote per candidate (mirroring ``confirm_findings`` with a
single-member roster) — and grades the artifact of the whole pipeline. What
ships is the confirmed set, so that is what gets the number; the candidate
list rides along so the grader can localize a miss to finder vs verify.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from agents.evals.config import settings
from agents.evals.fixtures import load_goldsets, read_input
from agents.evals.graders.decomposition import grade_boundedness, grade_decompose
from agents.evals.graders.replan import grade as grade_replan
from agents.evals.graders.review import grade_confirm, grade_findings
from agents.evals.graders.testgap import grade_find, grade_skeptic
from agents.evals.models import (
    CaseScore,
    GoldCase,
    GradeResult,
    Scorecard,
    StepScore,
)
from agents.evals.steps import ADAPTERS, PromptSpec, build_confirm_user
from agents.shared.ensemble import ApiExecutor, ExecResult, Executor, Pool, Prompt
from agents.shared.llm import extract_json

# ---------------------------------------------------------------------------
# Grader registry: 7 step keys → grading functions
# ---------------------------------------------------------------------------

GRADERS: dict[str, Callable[[GoldCase, str], GradeResult]] = {
    "replan": grade_replan,
    "decompose": grade_decompose,
    "boundedness": grade_boundedness,
    "review-findings": grade_findings,
    "review-confirm": grade_confirm,
    "testgap-find": grade_find,
    "testgap-skeptic": grade_skeptic,
}


def _build_executor(
    model: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
) -> ApiExecutor:
    """Build a single ApiExecutor for the given model."""
    return ApiExecutor(
        label=f"evals:{model}",
        kind="openai",
        model=model,
        base_url=base_url or settings.openai_base_url,
        api_key=api_key or settings.openai_api_key,
    )


def _call(pool: Pool, system: str, user: str, temp: float, timeout: float) -> ExecResult:
    """One raw model call — transport retries only, NO validate callback."""
    prompt = Prompt(system=system, user=user, max_tokens=settings.max_tokens, temperature=temp)
    return asyncio.run(pool.run(prompt, timeout=timeout, validate=None))


_SEVERITY_ORDER = ["critical", "high", "medium", "low"]


def _run_review_pipeline(
    case: GoldCase, pool: Pool, temp: float, timeout: float
) -> tuple[str | None, str | None]:
    """Run the finder→confirm funnel for one repeat.

    Returns ``(artifact_json, transport_error)`` — the artifact is what the
    review-findings grader consumes. Mirrors production:
    ``collect_findings`` (parse envelope, slug-dedup, severity-sort, cap) and
    ``confirm_findings`` (one vote per candidate; an unparseable or errored
    vote fails closed to not-real, like zero responders).
    """
    from agents.coding_pipeline.verify import CONFIRM_SYSTEM, FindingsEnvelope, stable_slug

    spec = ADAPTERS["review-findings"].build(case)
    finder_res = _call(pool, spec.system, spec.user, temp, timeout)
    if not finder_res.ok:
        return None, finder_res.error or "finder call failed"
    finder_raw = finder_res.output

    finder_valid = False
    candidates: list[dict] = []
    data = extract_json(finder_raw)
    if data:
        try:
            envelope = FindingsEnvelope.model_validate(data)
        except ValidationError:
            envelope = None
        if envelope is not None:
            finder_valid = True
            by_slug: dict[str, dict] = {}
            for f in envelope.findings:
                severity = f.severity if f.severity in _SEVERITY_ORDER else "medium"
                slug = stable_slug(f.file, f.summary)
                by_slug.setdefault(
                    slug,
                    {"slug": slug, "summary": f.summary, "file": f.file, "severity": severity},
                )
            candidates = sorted(
                by_slug.values(), key=lambda c: _SEVERITY_ORDER.index(c["severity"])
            )[: settings.review_max_candidates]

    diff = read_input(case, "diff.patch")
    confirmed: list[dict] = []
    votes: list[dict] = []
    confirm_errors = 0
    for cand in candidates:
        user = build_confirm_user(diff, cand["summary"], cand["file"], cand["severity"])
        vote_res = _call(pool, CONFIRM_SYSTEM, user, temp, timeout)
        if not vote_res.ok:
            confirm_errors += 1
            votes.append({"slug": cand["slug"], "real": False, "error": vote_res.error})
            continue
        vote_data = extract_json(vote_res.output)
        real = bool(vote_data.get("real") is True) if isinstance(vote_data, dict) else False
        votes.append({"slug": cand["slug"], "real": real})
        if real:
            confirmed.append(cand)

    artifact = json.dumps(
        {
            "finder_valid": finder_valid,
            "finder_raw": finder_raw,
            "candidates": candidates,
            "confirmed": confirmed,
            "votes": votes,
            "confirm_errors": confirm_errors,
        }
    )
    return artifact, None


def run_scorecard(
    model: str,
    *,
    steps: list[str] | None = None,
    goldsets_root: Path | None = None,
    repeats: int | None = None,
    temperature: float | None = None,
    executor_factory: Callable[[], Executor] | None = None,
) -> Scorecard:
    """Run a scorecard for *model* across all gold cases.

    Parameters
    ----------
    model:
        Model identifier used to build the executor.
    steps:
        If given, only run cases whose step is in this list.
    goldsets_root:
        Override the goldsets directory. Defaults to ``settings.goldsets_dir``.
    repeats:
        Number of repeats per case. Defaults to ``settings.repeats``.
    temperature:
        Pin temperature for determinism. Defaults to ``settings.temperature``.
    executor_factory:
        Test seam: return an ``Executor`` instead of building a live one.

    Returns
    -------
    Scorecard
        Aggregated results with per-step breakdowns.
    """
    goldsets_root = goldsets_root or settings.goldsets_dir
    repeats = repeats if repeats is not None else settings.repeats
    temp = temperature if temperature is not None else settings.temperature

    # Load gold cases, optionally filtered by step
    all_cases = load_goldsets(goldsets_root)
    if steps:
        all_cases = [c for c in all_cases if c.step in steps]

    # Build pool: single executor, no validation callback
    if executor_factory is not None:
        executor = executor_factory()
    else:
        executor = _build_executor(model)

    pool = Pool(role=f"evals:{model}", executors=[executor])

    timeout = settings.timeout

    # Run each case x repeats
    step_scores: dict[str, StepScore] = {}

    for case in all_cases:
        repeats_list: list[GradeResult] = []
        for _ in range(repeats):
            if case.step == "review-findings":
                # Full production funnel: finder -> dedup/cap -> confirm votes.
                raw_output, transport_error = _run_review_pipeline(case, pool, temp, timeout)
            else:
                adapter = ADAPTERS[case.step]
                spec: PromptSpec = adapter.build(case)
                # CRITICAL: NO validate callback — evals must grade every
                # attempt as-is, not retry on schema failure.
                result = _call(pool, spec.system, spec.user, temp, timeout)
                raw_output = result.output if result.ok else None
                transport_error = (
                    None if result.ok else result.error or f"transport: {result.status.value}"
                )

            if raw_output is not None:
                grader = GRADERS.get(case.step)
                if grader is None:
                    grade_result = GradeResult(
                        case_id=case.case_id,
                        step=case.step,  # type: ignore[arg-type]
                        passed=False,
                        score=0.0,
                        error=f"no grader for step {case.step}",
                    )
                else:
                    grade_result = grader(case, raw_output)
            else:
                # Transport failure: record error GradeResult
                grade_result = GradeResult(
                    case_id=case.case_id,
                    step=case.step,  # type: ignore[arg-type]
                    passed=False,
                    score=0.0,
                    error=transport_error or "transport failed",
                )

            repeats_list.append(grade_result)

        # Aggregate into StepScore
        if case.step not in step_scores:
            step_scores[case.step] = StepScore(step=case.step)
        step_scores[case.step].cases.append(
            CaseScore(
                case_id=case.case_id,
                holdout=case.holdout,
                repeats=repeats_list,
            )
        )

    # Assemble Scorecard
    return Scorecard(
        model=model,
        timestamp=datetime.now(UTC).isoformat(),
        steps=list(step_scores.values()),
    )
