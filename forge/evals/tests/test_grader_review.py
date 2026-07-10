"""Tests for the review-findings and review-confirm graders.

Covers all acceptance criteria:

review-findings:
1. perfect match             -- all must_find matched => recall/precision 1.0
2. paraphrase via keywords   -- different wording, same keywords => matched
3. miss drops recall         -- missing a finding drops recall below floor
4. hallucinated finding      -- extra candidate drops precision
5. greedy matching           -- no double-counting
6. empty/empty edge          -- no must_find, no candidates => precision 1.0
7. invalid JSON              => score 0

review-confirm:
8. right verdict (real=True) => passed
9. right verdict (real=False) => passed
10. wrong verdict => failed

Shape:
11. GradeResult has all required fields
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.evals.graders.review import grade_confirm, grade_findings
from forge.evals.models import GoldCase, GradeCheck, GradeResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gold_case(
    case_id: str = "test-review",
    expected: dict | None = None,
) -> GoldCase:
    return GoldCase(
        step="review-findings",
        case_id=case_id,
        case_dir=Path("/tmp/evals/fake-case"),
        schema_version=1,
        expected=expected or {},
    )


def _make_confirm_gold_case(
    expected_real: bool = True,
) -> GoldCase:
    return GoldCase(
        step="review-confirm",
        case_id="test-confirm",
        case_dir=Path("/tmp/evals/fake-case"),
        schema_version=1,
        expected={"real": expected_real},
    )


def _artifact(findings_raw: str, confirmed: list | None = None) -> str:
    """Wrap an old-style finder envelope string into a runner pipeline artifact.

    By default the confirmed set equals the candidate set (a confirm stage that
    approves everything) so matching-semantics tests grade the shipped set."""
    data = json.loads(findings_raw)
    candidates = [
        {
            "slug": f"slug-{i}",
            "summary": f.get("summary", ""),
            "file": f.get("file"),
            "severity": f.get("severity", "medium"),
        }
        for i, f in enumerate(data.get("findings", []))
    ]
    return json.dumps(
        {
            "finder_valid": True,
            "finder_raw": findings_raw,
            "candidates": candidates,
            "confirmed": candidates if confirmed is None else confirmed,
            "votes": [],
            "confirm_errors": 0,
        }
    )


def _invalid_finder_artifact(finder_raw: str) -> str:
    """Artifact for a finder call whose output never validated."""
    return json.dumps(
        {
            "finder_valid": False,
            "finder_raw": finder_raw,
            "candidates": [],
            "confirmed": [],
            "votes": [],
            "confirm_errors": 0,
        }
    )


# ---------------------------------------------------------------------------
# review-findings: parsing
# ---------------------------------------------------------------------------


def test_findings_invalid_finder_output_fails():
    """A finder whose output never validated fails finder-valid and stops."""
    case = _make_gold_case()
    result = grade_findings(case, _invalid_finder_artifact("not json {{{"))

    assert isinstance(result, GradeResult)
    assert result.passed is False
    assert result.score == pytest.approx(0.0)
    assert len(result.checks) == 1
    assert result.checks[0].name == "finder-valid"
    assert result.checks[0].passed is False


def test_findings_unreadable_artifact_fails():
    """Junk that is not a runner artifact at all fails finder-valid."""
    case = _make_gold_case()
    result = grade_findings(case, "not json {{{")

    assert result.passed is False
    assert result.checks[0].name == "finder-valid"
    assert result.checks[0].passed is False


def test_findings_valid_empty_envelope():
    """An empty candidate set from a valid finder passes finder-valid."""
    case = _make_gold_case()
    result = grade_findings(case, _artifact('{"findings": []}'))

    assert result.checks[0].name == "finder-valid"
    assert result.checks[0].passed is True


# ---------------------------------------------------------------------------
# review-findings: perfect match
# ---------------------------------------------------------------------------


def test_findings_perfect_match():
    """All must_find entries matched => recall=1.0, precision=1.0, score=1.0."""
    case = _make_gold_case(
        expected={
            "must_find": [
                {
                    "id": "ref-1",
                    "keywords_any": ["null pointer"],
                },
                {
                    "id": "ref-2",
                    "keywords_any": ["buffer overflow"],
                },
            ],
        }
    )
    raw = (
        '{"findings": ['
        '{"summary": "Null pointer dereference in parser", '
        '"file": "parser.c", "severity": "critical"},'
        '{"summary": "Buffer overflow in read function", '
        '"file": "io.c", "severity": "high"}'
        "]}"
    )
    result = grade_findings(case, _artifact(raw))

    assert result.passed is True
    assert result.score == pytest.approx(1.0)
    recall_check = [c for c in result.checks if c.name == "confirmed-recall"][0]
    precision_check = [c for c in result.checks if c.name == "confirmed-precision"][0]
    assert recall_check.passed is True
    assert precision_check.passed is True


# ---------------------------------------------------------------------------
# review-findings: paraphrase matched via keywords
# ---------------------------------------------------------------------------


def test_findings_paraphrase_matched_via_keywords():
    """Different wording but matching keywords => candidate matches reference."""
    case = _make_gold_case(
        expected={
            "must_find": [
                {
                    "id": "ref-1",
                    "keywords_any": ["null pointer", "dereference"],
                },
            ],
        }
    )
    # Paraphrased finding uses different words but contains "pointer" and "dereference"
    raw = (
        '{"findings": ['
        '{"summary": "Pointer accessed after being set to null, dereference may occur", '
        '"file": "auth.c", "severity": "high"}'
        "]}"
    )
    result = grade_findings(case, _artifact(raw))

    recall_check = [c for c in result.checks if c.name == "confirmed-recall"][0]
    assert recall_check.passed is True
    assert "1/1" in recall_check.detail


def test_findings_keywords_all_match():
    """keywords_all: all substrings must appear."""
    case = _make_gold_case(
        expected={
            "must_find": [
                {
                    "id": "ref-1",
                    "keywords_all": ["session", "token", "expiry"],
                },
            ],
        }
    )
    raw = (
        '{"findings": ['
        '{"summary": "Session token expiry not enforced in handler", '
        '"file": "auth/handler.py", "severity": "medium"}'
        "]}"
    )
    result = grade_findings(case, _artifact(raw))

    recall_check = [c for c in result.checks if c.name == "confirmed-recall"][0]
    assert recall_check.passed is True


def test_findings_keywords_all_miss():
    """keywords_all: missing one substring => no match."""
    case = _make_gold_case(
        expected={
            "must_find": [
                {
                    "id": "ref-1",
                    "keywords_all": ["session", "token", "expiry"],
                },
            ],
        }
    )
    # Missing "expiry"
    raw = (
        '{"findings": ['
        '{"summary": "Session token validation missing", '
        '"file": "auth/handler.py", "severity": "medium"}'
        "]}"
    )
    result = grade_findings(case, _artifact(raw))

    recall_check = [c for c in result.checks if c.name == "confirmed-recall"][0]
    assert recall_check.passed is False


# ---------------------------------------------------------------------------
# review-findings: miss drops recall below floor
# ---------------------------------------------------------------------------


def test_findings_miss_drops_recall_below_floor():
    """Missing a required finding drops recall below recall_min (default 0.7)."""
    case = _make_gold_case(
        expected={
            "must_find": [
                {"id": "ref-1", "keywords_any": ["null pointer"]},
                {"id": "ref-2", "keywords_any": ["buffer overflow"]},
                {"id": "ref-3", "keywords_any": ["SQL injection"]},
            ],
        }
    )
    # Only catches 1 of 3 => recall = 0.333 < 0.7
    raw = (
        '{"findings": ['
        '{"summary": "Null pointer in auth module", "file": "auth.c", "severity": "high"}'
        "]}"
    )
    result = grade_findings(case, _artifact(raw))

    recall_check = [c for c in result.checks if c.name == "confirmed-recall"][0]
    assert recall_check.passed is False


# ---------------------------------------------------------------------------
# review-findings: hallucinated finding drops precision
# ---------------------------------------------------------------------------


def test_findings_hallucinated_finding_drops_precision():
    """Extra candidate with no matching reference drops precision."""
    case = _make_gold_case(
        expected={
            "must_find": [
                {"id": "ref-1", "keywords_any": ["null pointer"]},
            ],
        }
    )
    # 1 match + 1 hallucinated => precision = 0.5
    raw = (
        '{"findings": ['
        '{"summary": "Null pointer in parser", "file": "parser.c", "severity": "critical"},'
        '{"summary": "Unused variable in utils", "file": "utils.c", "severity": "low"}'
        "]}"
    )
    result = grade_findings(case, _artifact(raw))

    precision_check = [c for c in result.checks if c.name == "confirmed-precision"][0]
    assert "0.50" in precision_check.detail
    # precision 0.5 >= 0.5 (default precision_min) => passes threshold
    assert precision_check.passed is True


# ---------------------------------------------------------------------------
# review-findings: greedy matching never double-counts
# ---------------------------------------------------------------------------


def test_findings_greedy_no_double_count():
    """One candidate can only satisfy one reference (greedy best match)."""
    case = _make_gold_case(
        expected={
            "must_find": [
                {"id": "ref-1", "keywords_any": ["memory"]},
                {"id": "ref-2", "keywords_any": ["leak"]},
            ],
        }
    )
    # One candidate matches both keywords but should only be matched once
    raw = (
        '{"findings": ['
        '{"summary": "Memory leak detected in allocator", "file": "alloc.c", "severity": "high"},'
        '{"summary": "Null pointer in parser", "file": "parser.c", "severity": "critical"}'
        "]}"
    )
    result = grade_findings(case, _artifact(raw))

    # Candidate 0 matches both refs (memory, leak). Greedy assigns it to ref-1 (tie, first wins).
    # Candidate 1 matches neither (no "leak").
    # recall = 1/2 = 0.5 (only ref-1 matched), precision = 1/2 = 0.5 (only 1 candidate matched)
    recall_check = [c for c in result.checks if c.name == "confirmed-recall"][0]
    precision_check = [c for c in result.checks if c.name == "confirmed-precision"][0]
    assert "1/2" in recall_check.detail
    assert "1/2" in precision_check.detail


def test_findings_greedy_candidate_claimed_by_one_ref_only():
    """When one candidate is the best match for two refs, only one is claimed."""
    case = _make_gold_case(
        expected={
            "must_find": [
                {"id": "ref-1", "keywords_any": ["memory"]},
                {"id": "ref-2", "keywords_any": ["memory"]},
            ],
        }
    )
    # One candidate mentions "memory" but can only satisfy one ref
    raw = (
        '{"findings": ['
        '{"summary": "Memory corruption found", "file": "main.c", "severity": "high"}'
        "]}"
    )
    result = grade_findings(case, _artifact(raw))

    # recall = 1/2 = 0.5 (only one ref can be matched to the single candidate)
    recall_check = [c for c in result.checks if c.name == "confirmed-recall"][0]
    assert "1/2" in recall_check.detail
    assert recall_check.passed is False


# ---------------------------------------------------------------------------
# review-findings: empty/empty edge case
# ---------------------------------------------------------------------------


def test_findings_empty_empty_edge():
    """No must_find and no candidates => precision 1.0, F1 should be 1.0."""
    case = _make_gold_case(expected={"must_find": []})
    result = grade_findings(case, _artifact('{"findings": []}'))

    precision_check = [c for c in result.checks if c.name == "confirmed-precision"][0]
    assert "no candidates and no references" in precision_check.detail
    assert result.score == pytest.approx(1.0)
    assert result.passed is True


def test_findings_empty_must_find_with_candidates():
    """No must_find but candidate findings exist => precision 0."""
    case = _make_gold_case(expected={"must_find": []})
    raw = '{"findings": [{"summary": "Some finding", "file": "x.c", "severity": "low"}]}'
    result = grade_findings(case, _artifact(raw))

    precision_check = [c for c in result.checks if c.name == "confirmed-precision"][0]
    # With no references but candidates present, precision = 0
    assert "0.00" in precision_check.detail
    assert precision_check.passed is False


# ---------------------------------------------------------------------------
# review-findings: file_pattern matching
# ---------------------------------------------------------------------------


def test_findings_file_pattern_match():
    """Candidate file matches file_pattern regex => match."""
    case = _make_gold_case(
        expected={
            "must_find": [
                {
                    "id": "ref-1",
                    "file_pattern": r"auth\.(c|py)$",
                    "keywords_any": ["auth bug"],
                },
            ],
        }
    )
    raw = '{"findings": [{"summary": "Auth bug in handler", "file": "auth.c", "severity": "high"}]}'
    result = grade_findings(case, _artifact(raw))

    recall_check = [c for c in result.checks if c.name == "confirmed-recall"][0]
    assert recall_check.passed is True


def test_findings_file_pattern_no_match():
    """Candidate file does not match file_pattern => no match."""
    case = _make_gold_case(
        expected={
            "must_find": [
                {
                    "id": "ref-1",
                    "file_pattern": r"auth\.(c|py)$",
                    "keywords_any": ["auth bug"],
                },
            ],
        }
    )
    raw = (
        '{"findings": [{"summary": "Auth bug in handler", "file": "parser.c", "severity": "high"}]}'
    )
    result = grade_findings(case, _artifact(raw))

    recall_check = [c for c in result.checks if c.name == "confirmed-recall"][0]
    assert recall_check.passed is False


def test_findings_null_file_with_pattern_fails():
    """Candidate file is null but reference has file_pattern => no match."""
    case = _make_gold_case(
        expected={
            "must_find": [
                {
                    "id": "ref-1",
                    "file_pattern": r"auth\.py$",
                    "keywords_any": ["auth bug"],
                },
            ],
        }
    )
    raw = '{"findings": [{"summary": "Auth bug in handler", "file": null, "severity": "high"}]}'
    result = grade_findings(case, _artifact(raw))

    recall_check = [c for c in result.checks if c.name == "confirmed-recall"][0]
    assert recall_check.passed is False


def test_findings_null_file_without_pattern_ok():
    """Candidate file is null and no file_pattern => still matches on keywords."""
    case = _make_gold_case(
        expected={
            "must_find": [
                {
                    "id": "ref-1",
                    "keywords_any": ["auth bug"],
                },
            ],
        }
    )
    raw = '{"findings": [{"summary": "Auth bug in handler", "file": null, "severity": "high"}]}'
    result = grade_findings(case, _artifact(raw))

    recall_check = [c for c in result.checks if c.name == "confirmed-recall"][0]
    assert recall_check.passed is True


# ---------------------------------------------------------------------------
# review-findings: max-findings
# ---------------------------------------------------------------------------


def test_findings_max_findings_passes():
    """5 findings is within the limit."""
    case = _make_gold_case()
    findings = ",".join(
        [f'{{"summary": "finding {i}", "file": "x.c", "severity": "medium"}}' for i in range(5)]
    )
    raw = f'{{"findings": [{findings}]}}'
    result = grade_findings(case, _artifact(raw))

    max_check = [c for c in result.checks if c.name == "max-findings"][0]
    assert max_check.passed is True


def test_findings_max_findings_exceeded():
    """6 findings exceeds the cap."""
    case = _make_gold_case()
    findings = ",".join(
        [f'{{"summary": "finding {i}", "file": "x.c", "severity": "medium"}}' for i in range(6)]
    )
    raw = f'{{"findings": [{findings}]}}'
    result = grade_findings(case, _artifact(raw))

    max_check = [c for c in result.checks if c.name == "max-findings"][0]
    assert max_check.passed is False
    assert "6" in max_check.detail


# ---------------------------------------------------------------------------
# review-findings: severity-legal
# ---------------------------------------------------------------------------


def test_findings_severity_legal_all_pass():
    """All severity values are legal."""
    case = _make_gold_case()
    raw = (
        '{"findings": ['
        '{"summary": "f1", "file": "x.c", "severity": "critical"},'
        '{"summary": "f2", "file": "y.c", "severity": "high"},'
        '{"summary": "f3", "file": "z.c", "severity": "medium"},'
        '{"summary": "f4", "file": "w.c", "severity": "low"}'
        "]}"
    )
    result = grade_findings(case, _artifact(raw))

    sev_check = [c for c in result.checks if c.name == "severity-legal"][0]
    assert sev_check.passed is True


def test_findings_severity_illegal_fails():
    """An illegal severity value fails severity-legal."""
    case = _make_gold_case()
    raw = (
        '{"findings": ['
        '{"summary": "f1", "file": "x.c", "severity": "critical"},'
        '{"summary": "f2", "file": "y.c", "severity": "extreme"}'
        "]}"
    )
    result = grade_findings(case, _artifact(raw))

    sev_check = [c for c in result.checks if c.name == "severity-legal"][0]
    assert sev_check.passed is False
    assert "extreme" in sev_check.detail


# ---------------------------------------------------------------------------
# review-findings: score is F1
# ---------------------------------------------------------------------------


def test_findings_score_is_f1():
    """Score = F1 = 2 * (precision * recall) / (precision + recall)."""
    # recall = 1/2 = 0.5, precision = 2/3 = 0.667
    # F1 = 2 * 0.5 * 0.667 / (0.5 + 0.667) = 0.571
    case = _make_gold_case(
        expected={
            "must_find": [
                {"id": "ref-1", "keywords_any": ["bug a"]},
                {"id": "ref-2", "keywords_any": ["bug b"]},
            ],
        }
    )
    raw = (
        '{"findings": ['
        '{"summary": "Bug a found", "file": "a.c", "severity": "high"},'
        '{"summary": "Bug b found", "file": "b.c", "severity": "high"},'
        '{"summary": "Bug c found", "file": "c.c", "severity": "low"}'
        "]}"
    )
    result = grade_findings(case, _artifact(raw))

    recall_check = [c for c in result.checks if c.name == "confirmed-recall"][0]
    precision_check = [c for c in result.checks if c.name == "confirmed-precision"][0]
    # recall = 2/2 = 1.0, precision = 2/3 = 0.667
    assert "2/2" in recall_check.detail
    assert "2/3" in precision_check.detail
    expected_f1 = 2 * 1.0 * (2 / 3) / (1.0 + 2 / 3)
    assert result.score == pytest.approx(round(expected_f1, 4))


# ---------------------------------------------------------------------------
# review-confirm: right verdict (real=True)
# ---------------------------------------------------------------------------


def test_confirm_right_verdict_real_true():
    """Model says real=True, expected real=True => passed."""
    case = _make_confirm_gold_case(expected_real=True)
    result = grade_confirm(case, '{"real": true, "reason": "clear evidence in diff"}')

    assert result.passed is True
    assert result.score == pytest.approx(1.0)
    confirm_check = [c for c in result.checks if c.name == "confirm-verdict"][0]
    assert confirm_check.passed is True


def test_confirm_right_verdict_real_false():
    """Model says real=False, expected real=False => passed."""
    case = _make_confirm_gold_case(expected_real=False)
    result = grade_confirm(case, '{"real": false, "reason": "style nit only"}')

    assert result.passed is True
    assert result.score == pytest.approx(1.0)
    confirm_check = [c for c in result.checks if c.name == "confirm-verdict"][0]
    assert confirm_check.passed is True


# ---------------------------------------------------------------------------
# review-confirm: wrong verdict
# ---------------------------------------------------------------------------


def test_confirm_wrong_verdict_real_true():
    """Model says real=False when expected real=True => failed."""
    case = _make_confirm_gold_case(expected_real=True)
    result = grade_confirm(case, '{"real": false, "reason": "not convinced"}')

    assert result.passed is False
    assert result.score == pytest.approx(0.0)
    confirm_check = [c for c in result.checks if c.name == "confirm-verdict"][0]
    assert confirm_check.passed is False


def test_confirm_wrong_verdict_real_false():
    """Model says real=True when expected real=False => failed."""
    case = _make_confirm_gold_case(expected_real=False)
    result = grade_confirm(case, '{"real": true, "reason": "this is a bug"}')

    assert result.passed is False
    assert result.score == pytest.approx(0.0)
    confirm_check = [c for c in result.checks if c.name == "confirm-verdict"][0]
    assert confirm_check.passed is False


# ---------------------------------------------------------------------------
# review-confirm: parse failure
# ---------------------------------------------------------------------------


def test_confirm_invalid_json_fails():
    """Non-parseable output fails confirm-parse."""
    case = _make_confirm_gold_case()
    result = grade_confirm(case, "not json")

    assert result.passed is False
    assert result.score == pytest.approx(0.0)
    assert result.checks[0].name == "confirm-parse"
    assert result.checks[0].passed is False


def test_confirm_invalid_schema_fails():
    """JSON that parses but fails ConfirmVerdict schema fails."""
    case = _make_confirm_gold_case()
    # Missing required "real" field
    result = grade_confirm(case, '{"reason": "no real"}')

    assert result.passed is False
    assert result.checks[0].passed is False


# ---------------------------------------------------------------------------
# GradeResult shape
# ---------------------------------------------------------------------------


def test_findings_grade_result_shape():
    """GradeResult from grade_findings has all required fields."""
    case = _make_gold_case()
    result = grade_findings(case, "garbage")

    assert isinstance(result, GradeResult)
    assert result.case_id == "test-review"
    assert result.step == "review-findings"
    assert isinstance(result.checks, list)
    assert len(result.checks) >= 1
    assert isinstance(result.checks[0], GradeCheck)
    assert result.error is None


def test_confirm_grade_result_shape():
    """GradeResult from grade_confirm has all required fields."""
    case = _make_confirm_gold_case()
    result = grade_confirm(case, "garbage")

    assert isinstance(result, GradeResult)
    assert result.case_id == "test-confirm"
    assert result.step == "review-confirm"
    assert isinstance(result.checks, list)
    assert len(result.checks) >= 1
    assert isinstance(result.checks[0], GradeCheck)
    assert result.error is None


# ---------------------------------------------------------------------------
# Pipeline localization: finder vs verify
# ---------------------------------------------------------------------------


def test_findings_verify_kill_localized():
    """The finder surfaced the bug but the confirm vote killed it: the
    candidate-recall check passes (finder fine) while confirmed-recall fails
    (verify miscalibrated) — the split that tells prompt work where to aim."""
    case = _make_gold_case(
        expected={"must_find": [{"id": "ref-1", "keywords_any": ["null pointer"]}]}
    )
    raw = (
        '{"findings": [{"summary": "Null pointer dereference in parser", '
        '"file": "parser.c", "severity": "critical"}]}'
    )
    result = grade_findings(case, _artifact(raw, confirmed=[]))

    cand = [c for c in result.checks if c.name == "candidate-recall"][0]
    shipped = [c for c in result.checks if c.name == "confirmed-recall"][0]
    assert cand.passed is True
    assert shipped.passed is False
    assert result.passed is False
    assert result.score == pytest.approx(0.0)  # shipped F1 is what scores


def test_findings_finder_miss_fails_both_layers():
    """A finder miss fails candidate-recall AND confirmed-recall — worse than
    a verify kill, and the checks say so."""
    case = _make_gold_case(
        expected={"must_find": [{"id": "ref-1", "keywords_any": ["null pointer"]}]}
    )
    raw = '{"findings": [{"summary": "Style nit", "file": "a.c", "severity": "low"}]}'
    result = grade_findings(case, _artifact(raw, confirmed=[]))

    assert [c for c in result.checks if c.name == "candidate-recall"][0].passed is False
    assert [c for c in result.checks if c.name == "confirmed-recall"][0].passed is False
