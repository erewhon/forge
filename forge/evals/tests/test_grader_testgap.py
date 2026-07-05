"""Tests for the testgap-find and testgap-skeptic graders.

Covers all acceptance criteria:

testgap-find:
1. full-recall case         -- all must_find matched, no cry-wolf, good ordering
2. missed hole fails 1      -- recall drops below threshold
3. critical-severity trivia fails 3 -- cry-wolf high/critical that match nothing
4. inverted ordering fails 3 -- severity ranks reversed
5. <2 pairs skip-passes 3  -- no comparable pairs => severity-order = 1.0

testgap-skeptic:
6. both labels (real=True and real=False)

7. invalid JSON scores 0    -- for both find and skeptic

Shape:
8. GradeResult has all required fields
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.evals.graders.testgap import grade_find, grade_skeptic
from agents.evals.models import GoldCase, GradeCheck, GradeResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_find_gold_case(
    case_id: str = "test-find",
    expected: dict | None = None,
) -> GoldCase:
    return GoldCase(
        step="testgap-find",
        case_id=case_id,
        case_dir=Path("/tmp/evals/fake-case"),
        schema_version=1,
        expected=expected or {},
    )


def _make_skeptic_gold_case(
    expected_real: bool = True,
) -> GoldCase:
    return GoldCase(
        step="testgap-skeptic",
        case_id="test-skeptic",
        case_dir=Path("/tmp/evals/fake-case"),
        schema_version=1,
        expected={"real": expected_real},
    )


# ---------------------------------------------------------------------------
# testgap-find: full-recall case
# ---------------------------------------------------------------------------


def test_find_full_recall_passes_all_checks():
    """All must_find matched, no cry-wolf, good severity ordering => all 3 checks pass."""
    case = _make_find_gold_case(
        expected={
            "must_find": [
                {
                    "id": "ref-1",
                    "target_pattern": r"auth\.(py|js)$",
                    "keywords_any": ["authentication"],
                },
                {
                    "id": "ref-2",
                    "target_pattern": r"payment\.(py|js)$",
                    "keywords_any": ["payment"],
                },
                {
                    "id": "ref-3",
                    "target_pattern": r"api\.(py|js)$",
                    "keywords_any": ["rate limit"],
                },
            ],
            "nice_to_find": [],
        }
    )
    raw = (
        '{"gaps": ['
        '{"target": "auth.py", "gap_type": "unit", "why_it_matters": '
        '"Missing authentication tests in auth module", '
        '"suggested_test": "test_auth_flow", "severity": "medium"},'
        '{"target": "payment.py", "gap_type": "unit", "why_it_matters": '
        '"Payment processing has no edge case tests", '
        '"suggested_test": "test_payment_edge", "severity": "high"},'
        '{"target": "api.py", "gap_type": "integration", "why_it_matters": '
        '"Rate limiting is not tested under load", '
        '"suggested_test": "test_rate_limit", "severity": "critical"}'
        "]}"
    )
    result = grade_find(case, raw)

    assert result.passed is True
    assert result.score == pytest.approx(1.0)
    recall_check = [c for c in result.checks if c.name == "recall"][0]
    cw_check = [c for c in result.checks if c.name == "cry-wolf"][0]
    order_check = [c for c in result.checks if c.name == "severity-order"][0]
    assert recall_check.passed is True
    assert cw_check.passed is True
    assert order_check.passed is True


# ---------------------------------------------------------------------------
# testgap-find: missed hole fails (check 2)
# ---------------------------------------------------------------------------


def test_find_missed_hole_fails():
    """Missed must_find hole causes recall to drop below threshold => check 2 fails."""
    case = _make_find_gold_case(
        expected={
            "must_find": [
                {"id": "ref-1", "keywords_any": ["auth"]},
                {"id": "ref-2", "keywords_any": ["payment"]},
                {"id": "ref-3", "keywords_any": ["api"]},
            ],
        }
    )
    # Only matches 1 of 3 => recall = 0.333 < 0.7
    raw = (
        '{"gaps": ['
        '{"target": "auth.py", "gap_type": "unit", "why_it_matters": '
        '"Authentication tests missing", "suggested_test": "x", "severity": "high"}'
        "]}"
    )
    result = grade_find(case, raw)

    recall_check = [c for c in result.checks if c.name == "recall"][0]
    assert recall_check.passed is False
    assert "1/3" in recall_check.detail
    assert result.passed is False


# ---------------------------------------------------------------------------
# testgap-find: critical-severity trivia fails cry-wolf (check 3)
# ---------------------------------------------------------------------------


def test_find_critical_severity_trivia_fails_cry_wolf():
    """High/critical severity candidate matching nothing fails cry-wolf."""
    case = _make_find_gold_case(
        expected={
            "must_find": [
                {"id": "ref-1", "keywords_any": ["auth"]},
            ],
            "nice_to_find": [],
        }
    )
    # 1 match + 2 critical-severity trivia that match nothing
    raw = (
        '{"gaps": ['
        '{"target": "auth.py", "gap_type": "unit", "why_it_matters": '
        '"Auth tests missing", "suggested_test": "x", "severity": "high"},'
        '{"target": "trivial-1.py", "gap_type": "unit", "why_it_matters": '
        '"Style guide not followed", "suggested_test": "x", "severity": "critical"},'
        '{"target": "trivial-2.py", "gap_type": "unit", "why_it_matters": '
        '"Naming convention violation", "suggested_test": "x", "severity": "critical"}'
        "]}"
    )
    result = grade_find(case, raw)

    cw_check = [c for c in result.checks if c.name == "cry-wolf"][0]
    assert cw_check.passed is False
    assert "2 high/critical" in cw_check.detail
    assert result.passed is False


# ---------------------------------------------------------------------------
# testgap-find: inverted severity ordering fails (check 4)
# ---------------------------------------------------------------------------


def test_find_inverted_ordering_fails():
    """Severity ranks inverted compared to reference => severity-order fails."""
    case = _make_find_gold_case(
        expected={
            "must_find": [
                {"id": "ref-1", "keywords_any": ["auth"]},  # rank 0
                {"id": "ref-2", "keywords_any": ["payment"]},  # rank 1
                {"id": "ref-3", "keywords_any": ["api"]},  # rank 2
            ],
        }
    )
    # Inverted: ref-0 candidate is critical, ref-1 is medium, ref-2 is low
    # Reference expects low->high->critical (increasing severity with index)
    # Model gives critical->medium->low (decreasing severity)
    raw = (
        '{"gaps": ['
        '{"target": "auth.py", "gap_type": "unit", "why_it_matters": '
        '"Auth tests missing", "suggested_test": "x", "severity": "critical"},'
        '{"target": "payment.py", "gap_type": "unit", "why_it_matters": '
        '"Payment tests missing", "suggested_test": "x", "severity": "medium"},'
        '{"target": "api.py", "gap_type": "unit", "why_it_matters": '
        '"API tests missing", "suggested_test": "x", "severity": "low"}'
        "]}"
    )
    result = grade_find(case, raw)

    order_check = [c for c in result.checks if c.name == "severity-order"][0]
    assert order_check.passed is False


# ---------------------------------------------------------------------------
# testgap-find: <2 pairs skip-passes severity-order (check 4)
# ---------------------------------------------------------------------------


def test_find_less_than_2_pairs_skip_passes_order():
    """When there are < 2 comparable matched pairs, severity-order passes (score = 1.0)."""
    case = _make_find_gold_case(
        expected={
            "must_find": [
                {"id": "ref-1", "keywords_any": ["auth"]},
            ],
        }
    )
    raw = (
        '{"gaps": ['
        '{"target": "auth.py", "gap_type": "unit", "why_it_matters": '
        '"Auth tests missing", "suggested_test": "x", "severity": "high"}'
        "]}"
    )
    result = grade_find(case, raw)

    order_check = [c for c in result.checks if c.name == "severity-order"][0]
    assert order_check.passed is True
    assert "< 2 comparable" in order_check.detail


# ---------------------------------------------------------------------------
# testgap-find: invalid JSON scores 0
# ---------------------------------------------------------------------------


def test_find_invalid_json_scores_zero():
    """Invalid JSON raw output => score 0, check 1 fails."""
    case = _make_find_gold_case()
    result = grade_find(case, "not json {{{")

    assert result.passed is False
    assert result.score == pytest.approx(0.0)
    assert len(result.checks) == 1
    assert result.checks[0].name == "find-parse"
    assert result.checks[0].passed is False


def test_find_invalid_schema_scores_zero():
    """JSON that parses but is not a valid GapsEnvelope => score 0."""
    case = _make_find_gold_case()
    result = grade_find(case, '{"gaps": "not a list"}')

    assert result.passed is False
    assert result.score == pytest.approx(0.0)
    assert result.checks[0].passed is False


# ---------------------------------------------------------------------------
# testgap-find: recall with nice_to_find (they don't count for recall)
# ---------------------------------------------------------------------------


def test_find_nice_to_find_not_counted_for_recall():
    """nice_to_find entries are not counted in recall denominator."""
    case = _make_find_gold_case(
        expected={
            "must_find": [
                {"id": "ref-1", "keywords_any": ["auth"]},
            ],
            "nice_to_find": [
                {"id": "nice-1", "keywords_any": ["style"]},
                {"id": "nice-2", "keywords_any": ["format"]},
            ],
        }
    )
    # Matches only must_find, not nice_to_find
    raw = (
        '{"gaps": ['
        '{"target": "auth.py", "gap_type": "unit", "why_it_matters": '
        '"Auth tests missing", "suggested_test": "x", "severity": "high"},'
        '{"target": "style.py", "gap_type": "unit", "why_it_matters": '
        '"Style guide violations", "suggested_test": "x", "severity": "low"}'
        "]}"
    )
    result = grade_find(case, raw)

    recall_check = [c for c in result.checks if c.name == "recall"][0]
    # Only 1 must_find, matched => recall = 1.0
    assert recall_check.passed is True


# ---------------------------------------------------------------------------
# testgap-find: empty must_find, empty candidates
# ---------------------------------------------------------------------------


def test_find_empty_must_find_empty_candidates():
    """No must_find and no candidates => recall = 1.0, cry-wolf passes."""
    case = _make_find_gold_case(expected={"must_find": []})
    result = grade_find(case, '{"gaps": []}')

    recall_check = [c for c in result.checks if c.name == "recall"][0]
    cw_check = [c for c in result.checks if c.name == "cry-wolf"][0]
    assert recall_check.passed is True
    assert cw_check.passed is True


# ---------------------------------------------------------------------------
# testgap-skeptic: both labels (real=True and real=False)
# ---------------------------------------------------------------------------


def test_skeptic_real_true_passes():
    """Model says real=True, expected real=True => passed."""
    case = _make_skeptic_gold_case(expected_real=True)
    verdict = {
        "real": True,
        "confidence": "high",
        "severity": "critical",
        "reasoning": "clear evidence",
    }
    result = grade_skeptic(case, json.dumps(verdict))

    assert result.passed is True
    assert result.score == pytest.approx(1.0)
    verdict_check = [c for c in result.checks if c.name == "skeptic-verdict"][0]
    assert verdict_check.passed is True


def test_skeptic_real_false_passes():
    """Model says real=False, expected real=False => passed."""
    case = _make_skeptic_gold_case(expected_real=False)
    verdict = {
        "real": False,
        "confidence": "medium",
        "severity": "low",
        "reasoning": "already covered",
    }
    result = grade_skeptic(case, json.dumps(verdict))

    assert result.passed is True
    assert result.score == pytest.approx(1.0)
    verdict_check = [c for c in result.checks if c.name == "skeptic-verdict"][0]
    assert verdict_check.passed is True


def test_skeptic_wrong_verdict_fails():
    """Model says real=False when expected real=True => failed."""
    case = _make_skeptic_gold_case(expected_real=True)
    verdict = {
        "real": False,
        "confidence": "low",
        "severity": "low",
        "reasoning": "not real",
    }
    result = grade_skeptic(case, json.dumps(verdict))

    assert result.passed is False
    assert result.score == pytest.approx(0.0)
    verdict_check = [c for c in result.checks if c.name == "skeptic-verdict"][0]
    assert verdict_check.passed is False


# ---------------------------------------------------------------------------
# testgap-skeptic: invalid JSON scores 0
# ---------------------------------------------------------------------------


def test_skeptic_invalid_json_scores_zero():
    """Invalid JSON for skeptic => score 0."""
    case = _make_skeptic_gold_case()
    result = grade_skeptic(case, "not json at all")

    assert result.passed is False
    assert result.score == pytest.approx(0.0)
    assert len(result.checks) == 1
    assert result.checks[0].name == "skeptic-parse"
    assert result.checks[0].passed is False


def test_skeptic_invalid_schema_scores_zero():
    """JSON that parses but fails SkepticVerdict schema => score 0."""
    case = _make_skeptic_gold_case()
    result = grade_skeptic(case, '{"confidence": "high"}')  # missing required "real"

    assert result.passed is False
    assert result.score == pytest.approx(0.0)
    assert result.checks[0].passed is False


# ---------------------------------------------------------------------------
# GradeResult shape
# ---------------------------------------------------------------------------


def test_find_grade_result_shape():
    """GradeResult from grade_find has all required fields."""
    case = _make_find_gold_case()
    result = grade_find(case, "garbage")

    assert isinstance(result, GradeResult)
    assert result.case_id == "test-find"
    assert result.step == "testgap-find"
    assert isinstance(result.checks, list)
    assert len(result.checks) >= 1
    assert isinstance(result.checks[0], GradeCheck)
    assert result.error is None


def test_skeptic_grade_result_shape():
    """GradeResult from grade_skeptic has all required fields."""
    case = _make_skeptic_gold_case()
    result = grade_skeptic(case, "garbage")

    assert isinstance(result, GradeResult)
    assert result.case_id == "test-skeptic"
    assert result.step == "testgap-skeptic"
    assert isinstance(result.checks, list)
    assert len(result.checks) >= 1
    assert isinstance(result.checks[0], GradeCheck)
    assert result.error is None


# ---------------------------------------------------------------------------
# testgap-find: cry-wolf with low/medium severity matches nothing (should NOT fail)
# ---------------------------------------------------------------------------


def test_find_low_severity_trivia_does_not_fail_cry_wolf():
    """Low/medium severity candidates matching nothing don't count as cry-wolf."""
    case = _make_find_gold_case(
        expected={
            "must_find": [
                {"id": "ref-1", "keywords_any": ["auth"]},
            ],
            "nice_to_find": [],
        }
    )
    # Only 1 match + low-severity trivia => should pass cry-wolf
    raw = (
        '{"gaps": ['
        '{"target": "auth.py", "gap_type": "unit", "why_it_matters": '
        '"Auth tests missing", "suggested_test": "x", "severity": "high"},'
        '{"target": "trivial.py", "gap_type": "unit", "why_it_matters": '
        '"Naming convention", "suggested_test": "x", "severity": "low"}'
        "]}"
    )
    result = grade_find(case, raw)

    cw_check = [c for c in result.checks if c.name == "cry-wolf"][0]
    assert cw_check.passed is True


# ---------------------------------------------------------------------------
# testgap-find: severity-order with exact ordering passes
# ---------------------------------------------------------------------------


def test_find_severity_order_exact_match_passes():
    """Severity matches reference ordering => severity-order = 1.0."""
    case = _make_find_gold_case(
        expected={
            "must_find": [
                {"id": "ref-1", "keywords_any": ["auth"]},  # rank 0 = expected lowest
                {"id": "ref-2", "keywords_any": ["payment"]},  # rank 1
                {"id": "ref-3", "keywords_any": ["api"]},  # rank 2 = expected highest
            ],
        }
    )
    # Models severity in same order: low, medium, high
    raw = (
        '{"gaps": ['
        '{"target": "auth.py", "gap_type": "unit", "why_it_matters": '
        '"Auth tests", "suggested_test": "x", "severity": "low"},'
        '{"target": "payment.py", "gap_type": "unit", "why_it_matters": '
        '"Payment tests", "suggested_test": "x", "severity": "medium"},'
        '{"target": "api.py", "gap_type": "unit", "why_it_matters": '
        '"API tests", "suggested_test": "x", "severity": "high"}'
        "]}"
    )
    result = grade_find(case, raw)

    order_check = [c for c in result.checks if c.name == "severity-order"][0]
    assert order_check.passed is True
