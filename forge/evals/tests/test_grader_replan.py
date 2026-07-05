"""Tests for the replan step grader.

Covers all 6 checks against the contract in ``agents/evals/graders/replan``:

1. valid-envelope      -- parse + schema validation
2. must-actions         -- every must entry matched
3. no-forbidden         -- no forbidden kind or target
4. fixup-confirmed-only -- fixup finding_slugs are confirmed
5. no-extras            -- extra / require_empty enforcement
6. leaf-floors          -- estimate floor, requires_tests, content length
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from agents.evals.graders import grade
from agents.evals.models import GoldCase, GradeCheck, GradeResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_LONGBASE = "x" * 250  # content that passes the 200-char threshold


def _make_gold_case(
    case_dir: Path | None = None,
    expected: dict | None = None,
    inputs: dict | None = None,
) -> GoldCase:
    d = case_dir or Path("/tmp/evals/fake-case")
    return GoldCase(
        step="replan",
        case_id="test-replan",
        case_dir=d,
        schema_version=1,
        inputs=inputs or {},
        expected=expected or {},
    )


def _long_leaf() -> dict:
    return {
        "title": "wire the flux capacitor",
        "content": _LONGBASE,
        "feature": "Time Travel",
        "execution_mode": "Auto-OK",
        "requires_tests": True,
        "estimate": "s",
    }


# ---------------------------------------------------------------------------
# Check 1: valid-envelope
# ---------------------------------------------------------------------------


def test_valid_fixup_only_answer_passes():
    """A valid fixup-only answer with correct expected block passes all checks."""
    case_dir = Path(tempfile.mkdtemp())
    case = _make_gold_case(
        case_dir=case_dir,
        expected={
            "must": [{"kind": "fixup", "finding_slug": "bug-123"}],
        },
    )
    raw = json.dumps(
        {
            "actions": [
                {
                    "kind": "fixup",
                    "finding_slug": "bug-123",
                    "leaf": _long_leaf(),
                }
            ]
        }
    )
    result = grade(case, raw)

    assert result.passed is True
    assert result.score == pytest.approx(1.0)
    assert result.error is None
    # All 6 checks passed
    for check in result.checks:
        assert check.passed is True


def test_invalid_json_fails_check_1_only():
    """Non-parseable output fails check 1 only."""
    case = _make_gold_case(expected={})
    result = grade(case, "not json at all {{{")

    assert result.passed is False
    assert result.score == pytest.approx(0.0)
    assert len(result.checks) == 1
    assert result.checks[0].name == "valid-envelope"
    assert result.checks[0].passed is False


def test_invalid_schema_fails_check_1():
    """JSON that parses but fails ReplanEnvelope schema fails check 1."""
    case = _make_gold_case(expected={})
    # "actions" must be a list of discriminated union objects;
    # a dict with kind="fixup" but missing required "leaf" and "finding_slug" fails.
    raw = json.dumps({"actions": [{"kind": "fixup"}]})
    result = grade(case, raw)

    assert result.passed is False
    assert result.score == pytest.approx(0.0)
    assert len(result.checks) == 1
    assert result.checks[0].name == "valid-envelope"
    assert result.checks[0].passed is False


def test_valid_empty_actions_passes_all():
    """An empty actions list is valid (the prompt says it is)."""
    case = _make_gold_case(expected={"must": []})
    raw = json.dumps({"actions": []})
    result = grade(case, raw)

    assert result.passed is True
    assert result.score == pytest.approx(1.0)
    for check in result.checks:
        assert check.passed is True


# ---------------------------------------------------------------------------
# Check 3: no-forbidden
# ---------------------------------------------------------------------------


def test_touching_forbidden_target_fails_check_3():
    """An action targeting a forbidden leaf_title fails no-forbidden."""
    case = _make_gold_case(
        expected={
            "must": [{"kind": "fixup"}],
            "forbid_targets": ["wire the flux capacitor"],
        }
    )
    raw = json.dumps(
        {
            "actions": [
                {
                    "kind": "fixup",
                    "finding_slug": "bug-1",
                    "leaf": _long_leaf(),
                }
            ]
        }
    )
    result = grade(case, raw)

    no_forbidden = [c for c in result.checks if c.name == "no-forbidden"][0]
    assert no_forbidden.passed is False


def test_forbidden_kind_fails_check_3():
    """An action with a forbidden kind fails no-forbidden."""
    case = _make_gold_case(
        expected={
            "must": [],
            "forbid_kinds": ["escalate"],
        }
    )
    raw = json.dumps(
        {"actions": [{"kind": "escalate", "leaf_title": "old leaf", "diagnostics": "why"}]}
    )
    result = grade(case, raw)

    no_forbidden = [c for c in result.checks if c.name == "no-forbidden"][0]
    assert no_forbidden.passed is False


# ---------------------------------------------------------------------------
# Check 4: fixup-confirmed-only
# ---------------------------------------------------------------------------


def test_fixup_for_unconfirmed_slug_fails_check_4():
    """A fixup for a finding_slug not in report.json fails fixup-confirmed-only."""
    case_dir = Path(tempfile.mkdtemp())
    # Write a report.json with no confirmed findings
    report = {
        "findings": [
            {"slug": "other-bug", "confirmed": False},
            {"slug": "yet-other", "confirmed": False},
        ]
    }
    (case_dir / "report.json").write_text(json.dumps(report))

    case = _make_gold_case(
        case_dir=case_dir,
        expected={"must": [{"kind": "fixup"}]},
    )
    raw = json.dumps(
        {
            "actions": [
                {
                    "kind": "fixup",
                    "finding_slug": "not-confirmed",
                    "leaf": _long_leaf(),
                }
            ]
        }
    )
    result = grade(case, raw)

    fixup_check = [c for c in result.checks if c.name == "fixup-confirmed-only"][0]
    assert fixup_check.passed is False


def test_fixup_for_confirmed_slug_passes():
    """A fixup for a confirmed finding_slug passes fixup-confirmed-only."""
    case_dir = Path(tempfile.mkdtemp())
    report = {
        "findings": [
            {"slug": "real-bug", "confirmed": True},
        ]
    }
    (case_dir / "report.json").write_text(json.dumps(report))

    case = _make_gold_case(
        case_dir=case_dir,
        expected={"must": [{"kind": "fixup", "finding_slug": "real-bug"}]},
    )
    raw = json.dumps(
        {
            "actions": [
                {
                    "kind": "fixup",
                    "finding_slug": "real-bug",
                    "leaf": _long_leaf(),
                }
            ]
        }
    )
    result = grade(case, raw)

    fixup_check = [c for c in result.checks if c.name == "fixup-confirmed-only"][0]
    assert fixup_check.passed is True


# ---------------------------------------------------------------------------
# Check 5: no-extras
# ---------------------------------------------------------------------------


def test_extra_action_fails_check_5_without_allow_extra():
    """An extra unmatched action fails no-extras when allow_extra is false."""
    case = _make_gold_case(
        expected={
            "must": [{"kind": "fixup", "finding_slug": "bug-1"}],
            "allow_extra": False,
        }
    )
    raw = json.dumps(
        {
            "actions": [
                {
                    "kind": "fixup",
                    "finding_slug": "bug-1",
                    "leaf": _long_leaf(),
                },
                {
                    "kind": "escalate",
                    "leaf_title": "some leaf",
                    "diagnostics": "why",
                },
            ]
        }
    )
    result = grade(case, raw)

    extras_check = [c for c in result.checks if c.name == "no-extras"][0]
    assert extras_check.passed is False


def test_extra_action_passes_with_allow_extra():
    """Extra actions are skip-passed when allow_extra is true."""
    case = _make_gold_case(
        expected={
            "must": [{"kind": "fixup"}],
            "allow_extra": True,
        }
    )
    raw = json.dumps(
        {
            "actions": [
                {
                    "kind": "fixup",
                    "finding_slug": "bug-1",
                    "leaf": _long_leaf(),
                },
                {
                    "kind": "escalate",
                    "leaf_title": "some leaf",
                    "diagnostics": "why",
                },
            ]
        }
    )
    result = grade(case, raw)

    extras_check = [c for c in result.checks if c.name == "no-extras"][0]
    assert extras_check.passed is True


def test_require_empty_with_clean_wave_passes():
    """require_empty with an empty actions list passes."""
    case = _make_gold_case(expected={"require_empty": True})
    raw = json.dumps({"actions": []})
    result = grade(case, raw)

    extras_check = [c for c in result.checks if c.name == "no-extras"][0]
    assert extras_check.passed is True


def test_require_empty_with_actions_fails():
    """require_empty with non-empty actions fails."""
    case = _make_gold_case(expected={"require_empty": True})
    raw = json.dumps({"actions": [{"kind": "fixup", "finding_slug": "x", "leaf": _long_leaf()}]})
    result = grade(case, raw)

    extras_check = [c for c in result.checks if c.name == "no-extras"][0]
    assert extras_check.passed is False


# ---------------------------------------------------------------------------
# Check 6: leaf-floors
# ---------------------------------------------------------------------------


def test_leaf_estimate_l_floor_violation_fails():
    """A leaf with estimate 'l' fails leaf-floors (must be xs/s/m)."""
    case = _make_gold_case(
        expected={"must": [{"kind": "fixup"}]},
    )
    big_leaf = {**_long_leaf(), "estimate": "l"}
    raw = json.dumps(
        {
            "actions": [
                {
                    "kind": "fixup",
                    "finding_slug": "bug-1",
                    "leaf": big_leaf,
                }
            ]
        }
    )
    result = grade(case, raw)

    floors_check = [c for c in result.checks if c.name == "leaf-floors"][0]
    assert floors_check.passed is False
    assert "estimate" in floors_check.detail.lower()


def test_leaf_short_content_violation_fails():
    """A leaf with content < 200 chars fails leaf-floors."""
    case = _make_gold_case(
        expected={"must": [{"kind": "fixup"}]},
    )
    short_leaf = {**_long_leaf(), "content": "short"}
    raw = json.dumps(
        {
            "actions": [
                {
                    "kind": "fixup",
                    "finding_slug": "bug-1",
                    "leaf": short_leaf,
                }
            ]
        }
    )
    result = grade(case, raw)

    floors_check = [c for c in result.checks if c.name == "leaf-floors"][0]
    assert floors_check.passed is False
    assert "content" in floors_check.detail.lower()


def test_leaf_non_manual_no_tests_fails():
    """A non-Manual leaf without requires_tests=true fails leaf-floors."""
    case = _make_gold_case(
        expected={"must": [{"kind": "fixup"}]},
    )
    auto_leaf = {
        **_long_leaf(),
        "execution_mode": "Auto-OK",
        "requires_tests": False,
    }
    raw = json.dumps(
        {
            "actions": [
                {
                    "kind": "fixup",
                    "finding_slug": "bug-1",
                    "leaf": auto_leaf,
                }
            ]
        }
    )
    result = grade(case, raw)

    floors_check = [c for c in result.checks if c.name == "leaf-floors"][0]
    assert floors_check.passed is False
    assert "requires_tests" in floors_check.detail.lower()


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


def test_score_is_passed_over_total():
    """Score = passed checks / total applicable checks."""
    case = _make_gold_case(expected={})
    raw = "not json"
    result = grade(case, raw)

    assert result.score == pytest.approx(0.0)
    assert result.passed is False


def test_all_checks_pass_score_is_one():
    """When all checks pass, score is 1.0."""
    case = _make_gold_case(expected={"must": [{"kind": "fixup"}]})
    raw = json.dumps(
        {
            "actions": [
                {
                    "kind": "fixup",
                    "finding_slug": "bug-1",
                    "leaf": _long_leaf(),
                }
            ]
        }
    )
    result = grade(case, raw)

    assert result.score == pytest.approx(1.0)
    assert result.passed is True


# ---------------------------------------------------------------------------
# Must-actions (check 2)
# ---------------------------------------------------------------------------


def test_must_actions_unmatched_fails():
    """A must entry that no action matches causes check 2 to fail."""
    case = _make_gold_case(
        expected={
            "must": [{"kind": "respec", "leaf_title": "missing leaf"}],
        }
    )
    raw = json.dumps(
        {
            "actions": [
                {
                    "kind": "fixup",
                    "finding_slug": "bug-1",
                    "leaf": _long_leaf(),
                }
            ]
        }
    )
    result = grade(case, raw)

    must_check = [c for c in result.checks if c.name == "must-actions"][0]
    assert must_check.passed is False


def test_must_actions_matched_passes():
    """All must entries matched => check 2 passes."""
    case = _make_gold_case(
        expected={
            "must": [
                {"kind": "fixup", "finding_slug": "bug-1"},
            ],
        }
    )
    raw = json.dumps(
        {
            "actions": [
                {
                    "kind": "fixup",
                    "finding_slug": "bug-1",
                    "leaf": _long_leaf(),
                }
            ]
        }
    )
    result = grade(case, raw)

    must_check = [c for c in result.checks if c.name == "must-actions"][0]
    assert must_check.passed is True


# ---------------------------------------------------------------------------
# No extras: each must entry matched at most once (one action covers one must)
# ---------------------------------------------------------------------------


def test_multiple_must_entries_all_matched():
    """Multiple must entries all satisfied."""
    case = _make_gold_case(
        expected={
            "must": [
                {"kind": "fixup", "finding_slug": "bug-1"},
                {"kind": "respec", "leaf_title": "old spec"},
            ],
        }
    )
    raw = json.dumps(
        {
            "actions": [
                {
                    "kind": "fixup",
                    "finding_slug": "bug-1",
                    "leaf": _long_leaf(),
                },
                {
                    "kind": "respec",
                    "leaf_title": "old spec",
                    "revised": _long_leaf(),
                    "rationale": "too big",
                },
            ]
        }
    )
    result = grade(case, raw)

    must_check = [c for c in result.checks if c.name == "must-actions"][0]
    assert must_check.passed is True

    extras_check = [c for c in result.checks if c.name == "no-extras"][0]
    assert extras_check.passed is True


# ---------------------------------------------------------------------------
# Integration: grade returns correct GradeResult shape
# ---------------------------------------------------------------------------


def test_grade_result_shape():
    """GradeResult has all required fields."""
    case = _make_gold_case(expected={})
    raw = "garbage"
    result = grade(case, raw)

    assert isinstance(result, GradeResult)
    assert result.case_id == "test-replan"
    assert result.step == "replan"
    assert isinstance(result.checks, list)
    assert len(result.checks) == 1
    assert isinstance(result.checks[0], GradeCheck)
    assert result.error is None
