"""Tests for the decompose and boundedness step graders.

Covers structural checks, rubric DSL, and boundedness grading:

Decompose structural checks:
1. valid-tree      – JSON parses, TaskTree validates, >= 1 leaf
2. deps-resolve    – no unknown deps, no cycles (2 failures)
3. files-named     – every leaf names a file
4. estimates-bounded – estimate in {xs,s,m} or novel+Manual
5. auto-floors     – non-Manual: requires_tests=true, max_files>=3

Rubric DSL:
- require_leaf    (pass + fail)
- forbid_leaf     (pass + fail)
- max_leaves      (pass + fail)
- min_leaves      (pass + fail)
- require_dep     (pass + fail)
- require_manual  (pass + fail)
- unknown kind    -> EvalFixtureError

Boundedness:
- right verdict (worker_shaped match)
- wrong verdict (worker_shaped mismatch)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.evals.fixtures import EvalFixtureError
from forge.evals.graders.decomposition import (
    grade_boundedness,
    grade_decompose,
)
from forge.evals.models import GoldCase

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gold_case(
    step: str = "decompose",
    expected: dict | None = None,
    rubric: list[dict] | None = None,
) -> GoldCase:
    exp = expected or {}
    if rubric is not None:
        exp["rubric"] = rubric
    return GoldCase(
        step=step,  # type: ignore[arg-type]
        case_id="test-decompose",
        case_dir=Path("/tmp/evals/fake-case"),
        schema_version=1,
        expected=exp,
    )


def _ok_tree(
    rubric: list[dict] | None = None,
    extra_expected: dict | None = None,
) -> tuple[GoldCase, str]:
    """A valid tree with reasonable defaults."""
    leaves = [
        {
            "title": "implement feature alpha",
            "content": "Build the API endpoint for /api/v1/users. File: api/users.py",
            "feature": "Alpha",
            "execution_mode": "Auto-OK",
            "requires_tests": True,
            "max_files": 3,
            "estimate": "s",
        },
        {
            "title": "write tests for alpha",
            "content": "Test the /api/v1/users endpoint. File: tests/test_users.py",
            "feature": "Alpha",
            "depends_on": ["implement feature alpha"],
            "execution_mode": "Auto-OK",
            "requires_tests": True,
            "max_files": 5,
            "estimate": "xs",
        },
    ]
    raw = json.dumps({"leaves": leaves})

    expected = {"rubric": rubric or []}
    if extra_expected:
        expected.update(extra_expected)

    return _make_gold_case(expected=expected), raw


# ---------------------------------------------------------------------------
# Check 1: valid-tree
# ---------------------------------------------------------------------------


def test_valid_tree_passes_check_1():
    """A valid tree with leaves passes valid-tree."""
    case, raw = _ok_tree()
    result = grade_decompose(case, raw)
    assert result.checks[0].passed is True
    assert result.checks[0].name == "valid-tree"


def test_invalid_json_fails_check_1():
    """Non-parseable output fails valid-tree."""
    case = _make_gold_case()
    result = grade_decompose(case, "not json")
    assert result.checks[0].passed is False
    assert result.checks[0].name == "valid-tree"


def test_empty_tree_fails_check_1():
    """A tree with no leaves fails valid-tree."""
    case = _make_gold_case()
    result = grade_decompose(case, json.dumps({"leaves": []}))
    assert result.checks[0].passed is False
    assert "no leaves" in result.checks[0].detail


# ---------------------------------------------------------------------------
# Check 2: deps-resolve
# ---------------------------------------------------------------------------


def test_deps_resolve_unknown_dep_fails():
    """A leaf depends on an unknown title fails deps-resolve."""
    leaves = [
        {
            "title": "leaf-a",
            "content": "File: a.py",
            "feature": "F",
            "depends_on": ["nonexistent-leaf"],
            "execution_mode": "Auto-OK",
            "requires_tests": True,
            "max_files": 3,
            "estimate": "s",
        },
    ]
    case = _make_gold_case()
    result = grade_decompose(case, json.dumps({"leaves": leaves}))
    deps_check = [c for c in result.checks if c.name == "deps-resolve"][0]
    assert deps_check.passed is False
    assert "unknown titles" in deps_check.detail


def test_deps_resolve_cycle_fails():
    """A cycle among leaf deps fails deps-resolve."""
    leaves = [
        {
            "title": "leaf-a",
            "content": "File: a.py",
            "feature": "F",
            "depends_on": ["leaf-b"],
            "execution_mode": "Auto-OK",
            "requires_tests": True,
            "max_files": 3,
            "estimate": "s",
        },
        {
            "title": "leaf-b",
            "content": "File: b.py",
            "feature": "F",
            "depends_on": ["leaf-a"],
            "execution_mode": "Auto-OK",
            "requires_tests": True,
            "max_files": 3,
            "estimate": "s",
        },
    ]
    case = _make_gold_case()
    result = grade_decompose(case, json.dumps({"leaves": leaves}))
    deps_check = [c for c in result.checks if c.name == "deps-resolve"][0]
    assert deps_check.passed is False
    assert "cycle" in deps_check.detail


# ---------------------------------------------------------------------------
# Check 3: files-named
# ---------------------------------------------------------------------------


def test_pathless_content_fails_check_3():
    """A leaf with no file paths in content fails files-named."""
    leaves = [
        {
            "title": "bare leaf",
            "content": "Just some text about a thing, no file references at all",
            "feature": "F",
            "execution_mode": "Auto-OK",
            "requires_tests": True,
            "max_files": 3,
            "estimate": "s",
        },
    ]
    case = _make_gold_case()
    result = grade_decompose(case, json.dumps({"leaves": leaves}))
    files_check = [c for c in result.checks if c.name == "files-named"][0]
    assert files_check.passed is False


def test_files_named_section_passes():
    """A leaf with a Files section passes files-named."""
    leaves = [
        {
            "title": "section leaf",
            "content": "## Files\napi/users.py\ntests/test.py",
            "feature": "F",
            "execution_mode": "Auto-OK",
            "requires_tests": True,
            "max_files": 3,
            "estimate": "s",
        },
    ]
    case = _make_gold_case()
    result = grade_decompose(case, json.dumps({"leaves": leaves}))
    files_check = [c for c in result.checks if c.name == "files-named"][0]
    assert files_check.passed is True


def test_novel_manual_skips_files_named():
    """A novel+Manual leaf is exempt from files-named."""
    leaves = [
        {
            "title": "novel manual leaf",
            "content": "No file paths here at all",
            "feature": "F",
            "execution_mode": "Manual",
            "requires_tests": True,
            "max_files": 1,
            "estimate": "l",
            "complexity": "novel",
        },
    ]
    case = _make_gold_case()
    result = grade_decompose(case, json.dumps({"leaves": leaves}))
    files_check = [c for c in result.checks if c.name == "files-named"][0]
    assert files_check.passed is True


# ---------------------------------------------------------------------------
# Check 4: estimates-bounded
# ---------------------------------------------------------------------------


def test_xl_estimate_fails_check_4():
    """An 'xl' estimate on a non-novel leaf fails estimates-bounded."""
    leaves = [
        {
            "title": "big leaf",
            "content": "File: big.py",
            "feature": "F",
            "execution_mode": "Auto-OK",
            "requires_tests": True,
            "max_files": 10,
            "estimate": "xl",
        },
    ]
    case = _make_gold_case()
    result = grade_decompose(case, json.dumps({"leaves": leaves}))
    est_check = [c for c in result.checks if c.name == "estimates-bounded"][0]
    assert est_check.passed is False
    assert "xl" in est_check.detail


# ---------------------------------------------------------------------------
# Check 5: auto-floors
# ---------------------------------------------------------------------------


def test_auto_leaf_governed_fields_are_floored_before_grading():
    """Production-equivalent grading: max_files=1 and requires_tests=False on an
    Auto leaf are governance-owned — _apply_conservative_tags floors them before
    the check, so they never fail (they never ship that way either)."""
    leaves = [
        {
            "title": "small auto leaf",
            "content": "File: small.py",
            "feature": "F",
            "execution_mode": "Auto-OK",
            "requires_tests": False,
            "max_files": 1,
            "estimate": "s",
        },
    ]
    case = _make_gold_case()
    result = grade_decompose(case, json.dumps({"leaves": leaves}))
    floors_check = [c for c in result.checks if c.name == "auto-floors"][0]
    assert floors_check.passed is True


def test_auto_leaf_oversized_max_files_fails_check_5():
    """max_files > 5 is NOT governed (governance only floors upward) — real
    model signal, fails auto-floors."""
    leaves = [
        {
            "title": "sprawling auto leaf",
            "content": "File: big.py",
            "feature": "F",
            "execution_mode": "Auto-OK",
            "requires_tests": True,
            "max_files": 8,
            "estimate": "s",
        },
    ]
    case = _make_gold_case()
    result = grade_decompose(case, json.dumps({"leaves": leaves}))
    floors_check = [c for c in result.checks if c.name == "auto-floors"][0]
    assert floors_check.passed is False
    assert "> 5" in floors_check.detail


# ---------------------------------------------------------------------------
# Rubric: require_leaf
# ---------------------------------------------------------------------------


def test_rubric_require_leaf_passes():
    """A require_leaf rubric item matched by a leaf passes."""
    case, raw = _ok_tree(
        rubric=[{"id": "has-alpha", "kind": "require_leaf", "title_pattern": "alpha"}]
    )
    result = grade_decompose(case, raw)
    rubric_check = [c for c in result.checks if c.name == "has-alpha"][0]
    assert rubric_check.passed is True


def test_rubric_require_leaf_fails():
    """A require_leaf rubric item not matched by any leaf fails."""
    case, raw = _ok_tree(
        rubric=[{"id": "has-bravo", "kind": "require_leaf", "title_pattern": "bravo"}]
    )
    result = grade_decompose(case, raw)
    rubric_check = [c for c in result.checks if c.name == "has-bravo"][0]
    assert rubric_check.passed is False


# ---------------------------------------------------------------------------
# Rubric: forbid_leaf
# ---------------------------------------------------------------------------


def test_rubric_forbid_leaf_passes():
    """No leaf matches the forbid pattern => passes."""
    case, raw = _ok_tree(
        rubric=[{"id": "no-bravo", "kind": "forbid_leaf", "title_pattern": "bravo"}]
    )
    result = grade_decompose(case, raw)
    rubric_check = [c for c in result.checks if c.name == "no-bravo"][0]
    assert rubric_check.passed is True


def test_rubric_forbid_leaf_fails():
    """A leaf matches the forbid pattern => fails."""
    case, raw = _ok_tree(
        rubric=[{"id": "no-alpha", "kind": "forbid_leaf", "title_pattern": "alpha"}]
    )
    result = grade_decompose(case, raw)
    rubric_check = [c for c in result.checks if c.name == "no-alpha"][0]
    assert rubric_check.passed is False


# ---------------------------------------------------------------------------
# Rubric: max_leaves
# ---------------------------------------------------------------------------


def test_rubric_max_leaves_passes():
    """Leaf count within max => passes."""
    case, raw = _ok_tree(rubric=[{"id": "small-enough", "kind": "max_leaves", "n": 5}])
    result = grade_decompose(case, raw)
    rubric_check = [c for c in result.checks if c.name == "small-enough"][0]
    assert rubric_check.passed is True


def test_rubric_max_leaves_fails():
    """Leaf count exceeds max => fails."""
    case, raw = _ok_tree(rubric=[{"id": "too-many", "kind": "max_leaves", "n": 1}])
    result = grade_decompose(case, raw)
    rubric_check = [c for c in result.checks if c.name == "too-many"][0]
    assert rubric_check.passed is False


# ---------------------------------------------------------------------------
# Rubric: min_leaves
# ---------------------------------------------------------------------------


def test_rubric_min_leaves_passes():
    """Leaf count meets min => passes."""
    case, raw = _ok_tree(rubric=[{"id": "enough", "kind": "min_leaves", "n": 1}])
    result = grade_decompose(case, raw)
    rubric_check = [c for c in result.checks if c.name == "enough"][0]
    assert rubric_check.passed is True


def test_rubric_min_leaves_fails():
    """Leaf count below min => fails."""
    leaves = [{"title": "one leaf", "content": "File: x.py", "feature": "F"}]
    case = _make_gold_case(rubric=[{"id": "two-min", "kind": "min_leaves", "n": 2}])
    result = grade_decompose(case, json.dumps({"leaves": leaves}))
    rubric_check = [c for c in result.checks if c.name == "two-min"][0]
    assert rubric_check.passed is False


# ---------------------------------------------------------------------------
# Rubric: require_dep
# ---------------------------------------------------------------------------


def test_rubric_require_dep_passes():
    """A required dependency edge exists => passes."""
    case, raw = _ok_tree(
        rubric=[
            {
                "id": "has-dep",
                "kind": "require_dep",
                "from_pattern": "tests for alpha",
                "to_pattern": "implement feature alpha",
            }
        ]
    )
    result = grade_decompose(case, raw)
    rubric_check = [c for c in result.checks if c.name == "has-dep"][0]
    assert rubric_check.passed is True


def test_rubric_require_dep_fails():
    """No matching dependency edge => fails."""
    case, raw = _ok_tree(
        rubric=[
            {
                "id": "missing-dep",
                "kind": "require_dep",
                "from_pattern": "leaf-a",
                "to_pattern": "leaf-b",
            }
        ]
    )
    result = grade_decompose(case, raw)
    rubric_check = [c for c in result.checks if c.name == "missing-dep"][0]
    assert rubric_check.passed is False


# ---------------------------------------------------------------------------
# Rubric: require_manual
# ---------------------------------------------------------------------------


def test_rubric_require_manual_passes():
    """All matching leaves are Manual => passes."""
    leaves = [
        {
            "title": "manual leaf",
            "content": "File: manual.py",
            "feature": "F",
            "execution_mode": "Manual",
        },
    ]
    case = _make_gold_case(
        rubric=[{"id": "all-manual", "kind": "require_manual", "title_pattern": "manual"}]
    )
    result = grade_decompose(case, json.dumps({"leaves": leaves}))
    rubric_check = [c for c in result.checks if c.name == "all-manual"][0]
    assert rubric_check.passed is True


def test_rubric_require_manual_fails():
    """A matching leaf is not Manual => fails."""
    leaves = [
        {
            "title": "should-be-manual",
            "content": "File: a.py",
            "feature": "F",
            "execution_mode": "Auto-OK",
        },
    ]
    case = _make_gold_case(
        rubric=[{"id": "must-be-manual", "kind": "require_manual", "title_pattern": "should-be"}]
    )
    result = grade_decompose(case, json.dumps({"leaves": leaves}))
    rubric_check = [c for c in result.checks if c.name == "must-be-manual"][0]
    assert rubric_check.passed is False


# ---------------------------------------------------------------------------
# Unknown rubric kind -> EvalFixtureError
# ---------------------------------------------------------------------------


def test_unknown_rubric_kind_raises():
    """An unknown rubric kind raises EvalFixtureError."""
    case, raw = _ok_tree(rubric=[{"id": "weird", "kind": "nonexistent_kind"}])
    with pytest.raises(EvalFixtureError, match="unknown rubric kind"):
        grade_decompose(case, raw)


# ---------------------------------------------------------------------------
# Boundedness: parse and scoring
# ---------------------------------------------------------------------------


def _make_boundedness_case(
    worker_shaped: bool = True,
    criteria: dict | None = None,
) -> tuple[GoldCase, str]:
    """Create a boundedness case with a matching raw output."""
    raw = json.dumps(
        {
            "leaf_title": "my leaf",
            "single_concern": True,
            "bounded_diff": True,
            "small_estimate": True,
            "testable_acceptance": True,
            "files_named": True,
        }
    )
    expected: dict = {"worker_shaped": worker_shaped}
    if criteria:
        expected["criteria"] = criteria
    case = _make_gold_case(step="boundedness", expected=expected)
    return case, raw


def test_boundedness_right_verdict_passes():
    """worker_shaped matches expected => all checks pass."""
    case, raw = _make_boundedness_case(worker_shaped=True)
    result = grade_boundedness(case, raw)
    assert result.checks[0].passed is True  # parse
    ws_check = [c for c in result.checks if c.name == "worker_shaped"][0]
    assert ws_check.passed is True


def test_boundedness_wrong_verdict_fails():
    """worker_shaped does not match expected => fails."""
    case, raw = _make_boundedness_case(worker_shaped=False)
    result = grade_boundedness(case, raw)
    ws_check = [c for c in result.checks if c.name == "worker_shaped"][0]
    assert ws_check.passed is False


def test_boundedness_criteria_comparison():
    """Named criteria fields are compared against expected."""
    raw = json.dumps(
        {
            "leaf_title": "my leaf",
            "single_concern": True,
            "bounded_diff": False,
            "small_estimate": True,
            "testable_acceptance": True,
            "files_named": True,
        }
    )
    expected = {
        "worker_shaped": False,
        "criteria": {"bounded_diff": False},
    }
    case = _make_gold_case(step="boundedness", expected=expected)
    result = grade_boundedness(case, raw)
    ws_check = [c for c in result.checks if c.name == "worker_shaped"][0]
    assert ws_check.passed is True  # worker_shaped is False and actual is False
    crit_check = [c for c in result.checks if c.name == "criteria.bounded_diff"][0]
    assert crit_check.passed is True


def test_boundedness_criteria_mismatch():
    """A criteria field mismatch shows in the check."""
    raw = json.dumps(
        {
            "leaf_title": "my leaf",
            "single_concern": True,
            "bounded_diff": True,
            "small_estimate": True,
            "testable_acceptance": True,
            "files_named": True,
        }
    )
    expected = {
        "worker_shaped": True,
        "criteria": {"bounded_diff": False},
    }
    case = _make_gold_case(step="boundedness", expected=expected)
    result = grade_boundedness(case, raw)
    crit_check = [c for c in result.checks if c.name == "criteria.bounded_diff"][0]
    assert crit_check.passed is False


def test_boundedness_invalid_json_fails():
    """Non-JSON input fails boundedness parse."""
    case = _make_gold_case(step="boundedness", expected={"worker_shaped": True})
    result = grade_boundedness(case, "not json")
    assert result.checks[0].passed is False
    assert result.checks[0].name == "boundedness-parse"


# ---------------------------------------------------------------------------
# Score computation for decompose
# ---------------------------------------------------------------------------


def test_decompose_score_passed_over_total():
    """Score = passed checks / total checks for decompose."""
    case = _make_gold_case()
    result = grade_decompose(case, "garbage")
    assert result.checks[0].passed is False
    total = len(result.checks)
    passed = sum(1 for c in result.checks if c.passed)
    assert result.score == pytest.approx(passed / total)


def test_decompose_all_checks_pass():
    """When all checks pass, score is 1.0."""
    case, raw = _ok_tree()
    result = grade_decompose(case, raw)
    assert result.passed is True
    assert result.score == pytest.approx(1.0)


def test_decompose_with_rubric_includes_rubric_checks():
    """Rubric checks appear in the result after structural checks."""
    case, raw = _ok_tree(
        rubric=[
            {"id": "has-alpha", "kind": "require_leaf", "title_pattern": "alpha"},
        ]
    )
    result = grade_decompose(case, raw)
    names = [c.name for c in result.checks]
    assert "valid-tree" in names
    assert "deps-resolve" in names
    assert "files-named" in names
    assert "estimates-bounded" in names
    assert "auto-floors" in names
    assert "has-alpha" in names
    assert len(result.checks) == 6  # 5 structural + 1 rubric


def test_boundedness_score_passed_over_total():
    """Score = passed checks / total checks for boundedness."""
    case, raw = _make_boundedness_case(worker_shaped=False)
    result = grade_boundedness(case, raw)
    assert result.passed is False
    total = len(result.checks)
    passed = sum(1 for c in result.checks if c.passed)
    assert result.score == pytest.approx(passed / total)


# ---------------------------------------------------------------------------
# Integration: GradeResult shape
# ---------------------------------------------------------------------------


def test_grade_result_shape_decompose():
    """GradeResult from decompose has all required fields."""
    from forge.evals.models import GradeCheck, GradeResult

    case = _make_gold_case()
    result = grade_decompose(case, "garbage")
    assert isinstance(result, GradeResult)
    assert result.case_id == "test-decompose"
    assert result.step == "decompose"
    assert isinstance(result.checks, list)
    assert len(result.checks) >= 1
    assert isinstance(result.checks[0], GradeCheck)
    assert result.error is None


def test_grade_result_shape_boundedness():
    """GradeResult from boundedness has all required fields."""
    from forge.evals.models import GradeCheck, GradeResult

    case, raw = _make_boundedness_case()
    result = grade_boundedness(case, raw)
    assert isinstance(result, GradeResult)
    assert result.case_id == "test-decompose"
    assert result.step == "boundedness"
    assert isinstance(result.checks, list)
    assert len(result.checks) >= 1
    assert isinstance(result.checks[0], GradeCheck)
    assert result.error is None
