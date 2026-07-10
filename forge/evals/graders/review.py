"""Deterministic graders for the `review-findings` and `review-confirm` steps.

Review findings grading measures precision/recall against a frozen reference
finding set over a known-buggy diff.  Review confirm grading checks
skeptic-vote accuracy on labeled real/decoy candidates.

Contract
--------
``grade_findings(case, raw) -> GradeResult``
``grade_confirm(case, raw) -> GradeResult``
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from forge.evals.models import GoldCase, GradeCheck, GradeResult
from forge.shared.llm import extract_json

# ---------------------------------------------------------------------------
# Wire model for review-confirm
# ---------------------------------------------------------------------------


class _ConfirmVerdict(BaseModel):
    real: bool
    reason: str


# ---------------------------------------------------------------------------
# review-findings grader
# ---------------------------------------------------------------------------


def grade_findings(case: GoldCase, raw: str) -> GradeResult:
    """Grade one review PIPELINE artifact against *case*.

    ``raw`` is the runner's pipeline artifact (JSON): ``finder_valid`` (bool),
    ``finder_raw`` (str), ``candidates`` (finder output after the production
    slug-dedup/severity-cap), ``confirmed`` (what survived the confirm votes —
    the set that actually ships), ``confirm_errors`` (int).

    Expected block keys in ``case.expected``
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    * ``must_find`` -- list of dicts with:
        - ``id`` (str): reference finding identifier
        - ``file_pattern`` (str|None): regex to match against candidate ``file``
        - ``keywords_any`` (list[str]|None): at least one case-insensitive
          substring must appear in the candidate ``summary``
        - ``keywords_all`` (list[str]): all must appear case-insensitively
          in the candidate ``summary``
    * ``recall_min`` (float, default 0.7): minimum acceptable recall
    * ``precision_min`` (float, default 0.5): minimum acceptable precision

    Checks
    ^^^^^^
    1. ``finder-valid``       -- the finder call produced a valid envelope.
    2. ``candidate-recall``   -- FINDER recall vs must_find (localizes a miss
       to the finder even when the pipeline number also fails).
    3. ``confirmed-recall``   -- SHIPPED recall: the confirmed set vs must_find.
    4. ``confirmed-precision``-- SHIPPED precision of the confirmed set.
    5. ``max-findings``       -- finder respected the prompt's cap of 5.
    6. ``severity-legal``     -- candidate severities in the legal set.

    ``score`` = F1 of the CONFIRMED set (what ships); ``passed`` = all checks.
    """
    expected: dict[str, Any] = case.expected or {}
    must_find: list[dict[str, Any]] = expected.get("must_find") or []
    recall_min: float = expected.get("recall_min", 0.7)
    precision_min: float = expected.get("precision_min", 0.5)

    checks: list[GradeCheck] = []

    # -- Step 0: parse the pipeline artifact ----------------------------------
    artifact = _parse_artifact(raw)
    if artifact is None:
        checks.append(
            GradeCheck(name="finder-valid", passed=False, detail="unreadable pipeline artifact")
        )
        return _result(case.case_id, case.step, checks, f1=0.0)

    candidates: list[dict] = artifact.get("candidates") or []
    confirmed: list[dict] = artifact.get("confirmed") or []

    # -- Check 1: finder produced a valid envelope ----------------------------
    finder_valid = bool(artifact.get("finder_valid"))
    checks.append(
        GradeCheck(
            name="finder-valid",
            passed=finder_valid,
            detail="" if finder_valid else "finder output never validated as FindingsEnvelope",
        )
    )
    if not finder_valid:
        return _result(case.case_id, case.step, checks, f1=0.0)

    # -- Check 2: candidate (finder) recall — the localization layer ----------
    cand_recall, _, cand_recall_detail, _ = _compute_precision_recall(candidates, must_find)
    checks.append(
        GradeCheck(
            name="candidate-recall",
            passed=cand_recall >= recall_min,
            detail=f"finder: {cand_recall_detail} >= {recall_min}?",
        )
    )

    # -- Checks 3+4: shipped (confirmed) recall / precision --------------------
    recall, precision, recall_detail, precision_detail = _compute_precision_recall(
        confirmed, must_find
    )
    checks.append(
        GradeCheck(
            name="confirmed-recall",
            passed=recall >= recall_min,
            detail=f"shipped: {recall_detail} >= {recall_min}?",
        )
    )
    checks.append(
        GradeCheck(
            name="confirmed-precision",
            passed=precision >= precision_min,
            detail=f"shipped: {precision_detail} >= {precision_min}?",
        )
    )

    # -- Check 5: max-findings cap (finder discipline) -------------------------
    checks.append(_check_max_findings(candidates))

    # -- Check 6: severity values legal ----------------------------------------
    checks.append(_check_severity_legal(candidates))

    # Score = F1 of the confirmed (shipped) set.
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return _result(case.case_id, case.step, checks, f1=f1)


def _parse_artifact(raw: str) -> dict | None:
    """The pipeline artifact is runner-produced JSON; anything else is a bug."""
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _compute_precision_recall(
    candidates: list[Any],
    must_find: list[dict[str, Any]],
) -> tuple[float, float, str, str]:
    """Compute recall and precision from candidates vs references.

    Returns ``(recall, precision, recall_detail, precision_detail)``.

    One candidate may satisfy only one reference (greedy best match).
    """
    total_refs = len(must_find)
    matched_refs: set[int] = set()
    matched_cands: set[int] = set()

    for ci, cand in enumerate(candidates):
        best_ref_idx: int | None = None
        best_score = -1
        for ri, ref in enumerate(must_find):
            if ri in matched_refs:
                continue
            if _candidate_matches_ref(cand, ref):
                score = _match_score(cand, ref)
                if score > best_score:
                    best_score = score
                    best_ref_idx = ri
        if best_ref_idx is not None:
            matched_refs.add(best_ref_idx)
            matched_cands.add(ci)

    if total_refs == 0:
        recall = 1.0
    else:
        recall = len(matched_refs) / total_refs

    total_cands = len(candidates)
    if total_cands == 0:
        if total_refs == 0:
            precision = 1.0
        else:
            precision = 0.0
    else:
        precision = len(matched_cands) / total_cands

    if total_refs == 0:
        recall_detail = "no reference findings (recall = 1.0)"
    else:
        recall_detail = (
            f"matched {len(matched_refs)}/{total_refs} references (recall = {recall:.2f})"
        )

    if total_cands == 0:
        if total_refs == 0:
            precision_detail = "no candidates and no references (precision = 1.0)"
        else:
            precision_detail = "no candidates but references exist (precision = 0.0)"
    else:
        precision_detail = (
            f"matched {len(matched_cands)}/{total_cands} candidates (precision = {precision:.2f})"
        )

    return recall, precision, recall_detail, precision_detail


def _cand_get(cand: Any, key: str, default: Any = None) -> Any:
    """Candidates are dicts in the pipeline artifact; keep attr access for tests."""
    if isinstance(cand, dict):
        return cand.get(key, default)
    return getattr(cand, key, default)


def _candidate_matches_ref(cand: Any, ref: dict[str, Any]) -> bool:
    """Does *cand* satisfy a reference *ref*?

    File: candidate file must match file_pattern (regex), skipped when absent.
    Candidate file may be null only if pattern is absent.
    Keywords: case-insensitive substring matching on summary.
    """
    cand_file = _cand_get(cand, "file")
    cand_summary = _cand_get(cand, "summary", "")

    # File pattern check
    file_pattern = ref.get("file_pattern")
    if file_pattern:
        if cand_file is None:
            return False
        if not re.search(file_pattern, cand_file):
            return False
    else:
        # No pattern: candidate file may be null
        pass

    # Keywords check on summary
    keywords_any: list[str] | None = ref.get("keywords_any")
    if keywords_any:
        summary_lower = cand_summary.lower()
        if not any(kw.lower() in summary_lower for kw in keywords_any):
            return False

    keywords_all: list[str] | None = ref.get("keywords_all")
    if keywords_all:
        summary_lower = cand_summary.lower()
        if not all(kw.lower() in summary_lower for kw in keywords_all):
            return False

    return True


def _match_score(cand: Any, ref: dict[str, Any]) -> int:
    """Score for greedy matching: higher = better match."""
    score = 0
    cand_file = _cand_get(cand, "file")
    file_pattern = ref.get("file_pattern")
    if file_pattern and cand_file and re.search(file_pattern, cand_file):
        score += 10  # bonus for file match

    keywords_all: list[str] | None = ref.get("keywords_all")
    if keywords_all:
        summary_lower = _cand_get(cand, "summary", "").lower()
        score += len([kw for kw in keywords_all if kw.lower() in summary_lower])
    else:
        keywords_any: list[str] | None = ref.get("keywords_any")
        if keywords_any:
            summary_lower = _cand_get(cand, "summary", "").lower()
            score += len([kw for kw in keywords_any if kw.lower() in summary_lower])

    return score


def _check_max_findings(candidates: list[Any]) -> GradeCheck:
    """Check 3: prompt cap of 5 findings."""
    n = len(candidates)
    if n > 5:
        return GradeCheck(
            name="max-findings",
            passed=False,
            detail=f"found {n} findings (max 5)",
        )
    return GradeCheck(name="max-findings", passed=True)


def _check_severity_legal(candidates: list[Any]) -> GradeCheck:
    """Check 4: all severity values are legal."""
    legal = {"critical", "high", "medium", "low"}
    violations: list[str] = []
    for i, cand in enumerate(candidates):
        sev = _cand_get(cand, "severity")
        if sev not in legal:
            violations.append(f"finding {i}: severity '{sev}' not in {legal}")
    if violations:
        return GradeCheck(
            name="severity-legal",
            passed=False,
            detail="; ".join(violations),
        )
    return GradeCheck(name="severity-legal", passed=True)


# ---------------------------------------------------------------------------
# review-confirm grader
# ---------------------------------------------------------------------------


def grade_confirm(case: GoldCase, raw: str) -> GradeResult:
    """Grade a single review-confirm raw output against *case*.

    Parse ConfirmVerdict; compare ``real`` to ``expected.real`` (bool).
    Score 1/0; passed = correct.
    """
    expected: dict[str, Any] = case.expected or {}
    expected_real: bool = expected.get("real", False)

    # -- Step 1: parse verdict ------------------------------------------------
    parsed_ok = True
    verdict: _ConfirmVerdict | None = None
    try:
        data = extract_json(raw)
        if not data:
            parsed_ok = False
        else:
            verdict = _ConfirmVerdict.model_validate(data)
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
                    name="confirm-parse",
                    passed=False,
                    detail=("raw output is not valid JSON or schema"),
                ),
            ],
        )

    # -- Step 2: compare real to expected ------------------------------------
    correct = verdict.real == expected_real
    check = GradeCheck(
        name="confirm-verdict",
        passed=correct,
        detail=(
            f"expected real={expected_real}, actual real={verdict.real}, reason={verdict.reason}"
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
# Score computation
# ---------------------------------------------------------------------------


def _result(
    case_id: str,
    step: str,
    checks: list[GradeCheck],
    f1: float | None = None,
) -> GradeResult:
    """Build a GradeResult from a list of checks, applying thresholds
    for recall/precision on the review-findings step."""
    clean_checks: list[GradeCheck] = []
    for c in checks:
        clean_checks.append(GradeCheck(name=c.name, passed=c.passed, detail=c.detail))

    # When F1 is provided (findings grading), score is the shipped-set F1 —
    # but passed always means EVERY check (thresholds included) passed.
    if f1 is not None:
        return GradeResult(
            case_id=case_id,
            step=step,  # type: ignore[arg-type]
            passed=all(c.passed for c in clean_checks),
            score=round(f1, 4),
            checks=clean_checks,
        )

    applicable = clean_checks
    total = len(applicable)
    if total == 0:
        return GradeResult(
            case_id=case_id,
            step=step,  # type: ignore[arg-type]
            passed=True,
            score=1.0,
            checks=clean_checks,
        )

    passed_count = sum(1 for c in applicable if c.passed)
    score = round(passed_count / total, 4)
    all_passed = passed_count == total

    return GradeResult(
        case_id=case_id,
        step=step,  # type: ignore[arg-type]
        passed=all_passed,
        score=score,
        checks=clean_checks,
    )
