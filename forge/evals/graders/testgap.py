"""Deterministic graders for the `testgap-find` and `testgap-skeptic` steps.

Testgap-find grading measures whether the model finds real test gaps
(`recall`), avoids crying wolf on non-gaps (`cry-wolf`), and ranks
severity correctly (`severity-order`).  All three checks are combined
as a simple mean of their individual scores.

Testgap-skeptic grading compares the model's real/decoy verdicts
against labeled expected verdicts.

Contract
--------
``grade_find(case, raw) -> GradeResult``
``grade_skeptic(case, raw) -> GradeResult``
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

from agents.evals.models import GoldCase, GradeCheck, GradeResult
from agents.evals.steps import GapsEnvelope, SkepticVerdict, TestGapItem
from agents.shared.llm import extract_json

# ---------------------------------------------------------------------------
# Severity ordering helpers
# ---------------------------------------------------------------------------

_SEV_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1}


def _sev_rank(sev: str | None) -> int | None:
    """Convert a severity string to a numeric rank, or None if invalid/missing."""
    if sev is None:
        return None
    return _SEV_ORDER.get(sev.lower())


# ---------------------------------------------------------------------------
# testgap-find grader
# ---------------------------------------------------------------------------


def grade_find(case: GoldCase, raw: str) -> GradeResult:
    """Grade a single testgap-find raw output against *case*.

    Expected block keys in ``case.expected``
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    * ``must_find`` -- list of dicts with:
        - ``id`` (str): reference gap identifier
        - ``target_pattern`` (str|None): regex to match against candidate ``target``
        - ``keywords_any`` (list[str]|None): at least one case-insensitive
          substring must appear in the candidate ``why_it_matters``
        - ``keywords_all`` (list[str]): all must appear case-insensitively
          in the candidate ``why_it_matters``
    * ``nice_to_find`` -- list of dicts with:
        - ``id`` (str): reference gap identifier
        Matching rules same as above, but count is only for neither recall
        nor cry-wolf.
    * ``recall_min`` (float, default 0.7): minimum acceptable recall
    * ``cry_wolf_max`` (int, default 1): max allowed high|critical severity
      candidates that match NOTHING
    * ``order_min`` (float, default 0.6): minimum concordance rate for
      severity ordering

    Checks
    ------
    1. ``find-parse`` -- parses as GapsEnvelope
    2. ``recall`` -- matched must_find / must_find >= recall_min
    3. ``cry-wolf`` -- high|critical severity candidates matching nothing <= cry_wolf_max
    4. ``severity-order`` -- pairwise concordance >= order_min

    Score = mean of checks 2, 3, 4 (each ratio capped at 1).
    """
    expected: dict[str, Any] = case.expected or {}
    must_find: list[dict[str, Any]] = expected.get("must_find") or []
    nice_to_find: list[dict[str, Any]] = expected.get("nice_to_find") or []
    recall_min: float = expected.get("recall_min", 0.7)
    cry_wolf_max: int = expected.get("cry_wolf_max", 1)
    order_min: float = expected.get("order_min", 0.6)

    checks: list[GradeCheck] = []

    # -- Step 1: parse envelope -----------------------------------------------
    check_parse = _check_parse_envelope(raw)
    checks.append(check_parse)

    if not check_parse.passed:
        return _result_find(case.case_id, case.step, checks)

    envelope: GapsEnvelope = check_parse._envelope  # type: ignore[attr-defined]
    candidates: list[TestGapItem] = envelope.gaps

    # -- Step 2: recall -------------------------------------------------------
    recall_score, recall_detail = _check_recall(candidates, must_find)
    recall_met = recall_score >= recall_min
    checks.append(
        GradeCheck(
            name="recall",
            passed=recall_met,
            detail=f"{recall_detail} >= {recall_min}?",
        )
    )

    # -- Step 3: cry-wolf -----------------------------------------------------
    cw_offenders, cw_detail = _check_cry_wolf(candidates, must_find, nice_to_find)
    cw_score = 1.0 if cw_offenders <= cry_wolf_max else 0.0
    cw_met = cw_offenders <= cry_wolf_max
    checks.append(
        GradeCheck(
            name="cry-wolf",
            passed=cw_met,
            detail=f"{cw_detail} <= {cry_wolf_max}?",
        )
    )

    # -- Step 4: severity-order -----------------------------------------------
    order_score, order_detail = _check_severity_order(candidates, must_find)
    order_met = order_score >= order_min
    checks.append(
        GradeCheck(
            name="severity-order",
            passed=order_met,
            detail=f"{order_detail} >= {order_min}?",
        )
    )

    # Score = mean of the three check scores (each capped at 1)
    score = (recall_score + cw_score + order_score) / 3.0
    score = min(score, 1.0)

    return GradeResult(
        case_id=case.case_id,
        step=case.step,  # type: ignore[arg-type]
        passed=recall_met and cw_met and order_met,
        score=round(score, 4),
        checks=[GradeCheck(name=c.name, passed=c.passed, detail=c.detail) for c in checks],
    )


def _check_parse_envelope(raw: str) -> GradeCheck:
    """Check 1: parses as GapsEnvelope (invalid -> fail)."""
    try:
        data = extract_json(raw)
        if not data:
            return GradeCheck(
                name="find-parse",
                passed=False,
                detail="raw output is not valid JSON",
            )
        envelope = GapsEnvelope.model_validate(data)
        check = GradeCheck(name="find-parse", passed=True)
        check._envelope = envelope  # type: ignore[attr-defined]
        return check
    except ValidationError as exc:
        return GradeCheck(
            name="find-parse",
            passed=False,
            detail=f"schema validation failed: {exc}",
        )


def _reference_matches(
    candidate: TestGapItem,
    ref: dict[str, Any],
) -> bool:
    """Does *candidate* match a reference *ref*?

    Same shape as the review grader:
    - target: regex match against ``target_pattern``
    - keywords: case-insensitive substring on ``why_it_matters``
    """
    # Target pattern check
    target_pattern = ref.get("target_pattern")
    if target_pattern:
        if not re.search(target_pattern, candidate.target):
            return False
    # keywords_any: at least one must appear
    keywords_any: list[str] | None = ref.get("keywords_any")
    if keywords_any:
        why_lower = candidate.why_it_matters.lower()
        if not any(kw.lower() in why_lower for kw in keywords_any):
            return False
    # keywords_all: all must appear
    keywords_all: list[str] | None = ref.get("keywords_all")
    if keywords_all:
        why_lower = candidate.why_it_matters.lower()
        if not all(kw.lower() in why_lower for kw in keywords_all):
            return False
    return True


def _candidate_matches_any_reference(
    candidate: TestGapItem,
    references: list[dict[str, Any]],
) -> bool:
    """Does the candidate match at least one reference entry?"""
    return any(_reference_matches(candidate, ref) for ref in references)


def _check_recall(
    candidates: list[TestGapItem],
    must_find: list[dict[str, Any]],
) -> tuple[float, str]:
    """Compute recall: matched must_find / total must_find."""
    total_refs = len(must_find)
    if total_refs == 0:
        return (1.0, "no must_find references (recall = 1.0)")

    matched: set[int] = set()
    for cand in candidates:
        for ri, ref in enumerate(must_find):
            if ri not in matched and _reference_matches(cand, ref):
                matched.add(ri)
                break  # one candidate satisfies at most one ref

    recall = len(matched) / total_refs
    return (
        min(recall, 1.0),
        f"matched {len(matched)}/{total_refs} must_find (recall = {recall:.2f})",
    )


def _check_cry_wolf(
    candidates: list[TestGapItem],
    must_find: list[dict[str, Any]],
    nice_to_find: list[dict[str, Any]],
) -> tuple[int, str]:
    """Count candidates with severity high|critical that match NOTHING.

    Returns (offender_count, detail_string).
    """
    all_refs = list(must_find) + list(nice_to_find)
    offenders = 0
    for cand in candidates:
        sev_rank = _sev_rank(cand.severity)
        if sev_rank is not None and sev_rank >= _SEV_ORDER["high"]:
            if not _candidate_matches_any_reference(cand, all_refs):
                offenders += 1

    return (offenders, f"{offenders} high/critical severity gaps matching nothing")


def _check_severity_order(
    candidates: list[TestGapItem],
    must_find: list[dict[str, Any]],
) -> tuple[float, str]:
    """Pairwise severity ordering agreement between model ranks and reference ranks.

    For each matched must_find pair, compare the severity rank of the
    candidate that matched it against the reference rank (position in
    must_find list, higher index = higher expected rank).

    Concordant + discordant = comparable pairs.
    Score = concordant / (concordant + discordant).
    Ties are skipped.
    Skip-passes (score = 1.0) when < 2 comparable pairs.
    """
    # Build matched pairs: (candidate, ref_index)
    matched: list[tuple[TestGapItem, int]] = []
    used_refs: set[int] = set()
    for cand in candidates:
        for ri, ref in enumerate(must_find):
            if ri not in used_refs and _reference_matches(cand, ref):
                matched.append((cand, ri))
                used_refs.add(ri)
                break

    if len(matched) < 2:
        # < 2 comparable pairs: skip-pass
        return (
            1.0,
            f"< 2 comparable matched pairs ({len(matched)}), skip-pass",
        )

    # Compute pairwise concordance
    concordant = 0
    discordant = 0
    for i in range(len(matched)):
        for j in range(i + 1, len(matched)):
            cand_i, ref_i = matched[i]
            cand_j, ref_j = matched[j]
            rank_i = ref_i
            rank_j = ref_j
            # Reference rank difference
            ref_diff = rank_j - rank_i
            if ref_diff == 0:
                continue  # tie in reference ranks => skip
            # Model severity rank difference
            model_rank_i = _sev_rank(cand_i.severity)
            model_rank_j = _sev_rank(cand_j.severity)
            if model_rank_i is None or model_rank_j is None:
                continue  # can't compare
            model_diff = model_rank_j - model_rank_i
            if model_diff == 0:
                continue  # tie in model ranks => skip
            if (ref_diff > 0 and model_diff > 0) or (ref_diff < 0 and model_diff < 0):
                concordant += 1
            else:
                discordant += 1

    total = concordant + discordant
    if total == 0:
        return (1.0, "no comparable pairs after filtering ties, skip-pass")

    order_score = concordant / total
    return (
        min(order_score, 1.0),
        f"concordant={concordant}, discordant={discordant} (order = {order_score:.2f})",
    )


# ---------------------------------------------------------------------------
# testgap-skeptic grader
# ---------------------------------------------------------------------------


def grade_skeptic(case: GoldCase, raw: str) -> GradeResult:
    """Grade a single testgap-skeptic raw output against *case*.

    Parse ``SkepticVerdict``; compare ``real`` to ``expected.real``.
    Score = fraction correct; passed = all correct.
    """
    expected: dict[str, Any] = case.expected or {}
    expected_real: bool = expected.get("real", False)

    # -- Step 1: parse verdict ------------------------------------------------
    parsed_ok = True
    verdict: SkepticVerdict | None = None
    try:
        data = extract_json(raw)
        if not data:
            parsed_ok = False
        else:
            verdict = SkepticVerdict.model_validate(data)
    except ValidationError:
        parsed_ok = False

    if not parsed_ok or verdict is None:
        return GradeResult(
            case_id=case.case_id,
            step=case.step,  # type: ignore[arg-type]
            passed=False,
            score=0.0,
            checks=[
                GradeCheck(
                    name="skeptic-parse",
                    passed=False,
                    detail="raw output is not valid JSON or schema",
                ),
            ],
        )

    # -- Step 2: compare real to expected ------------------------------------
    correct = verdict.real == expected_real
    check = GradeCheck(
        name="skeptic-verdict",
        passed=correct,
        detail=(
            f"expected real={expected_real}, actual real={verdict.real}, "
            f"confidence={verdict.confidence}, severity={verdict.severity}"
        ),
    )

    score = 1.0 if correct else 0.0
    return GradeResult(
        case_id=case.case_id,
        step=case.step,  # type: ignore[arg-type]
        passed=correct,
        score=score,
        checks=[check],
    )


# ---------------------------------------------------------------------------
# Score computation (find)
# ---------------------------------------------------------------------------


def _result_find(
    case_id: str,
    step: str,
    checks: list[GradeCheck],
) -> GradeResult:
    """Build a GradeResult from a list of checks when parsing failed."""
    clean_checks: list[GradeCheck] = []
    for c in checks:
        clean_checks.append(GradeCheck(name=c.name, passed=c.passed, detail=c.detail))

    return GradeResult(
        case_id=case_id,
        step=step,  # type: ignore[arg-type]
        passed=False,
        score=0.0,
        checks=clean_checks,
    )
