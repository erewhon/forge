"""Deterministic graders for the `decompose` and `boundedness` steps.

Decompose grading is structure + rubric checklist — never exact-tree match.
Boundedness grading compares worker_shaped + optional criteria fields.

Contract
--------
``grade_decompose(case, raw) -> GradeResult``
``grade_boundedness(case, raw) -> GradeResult``
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import ValidationError

# _validate_deps from the architect module; raises ArchitectError on
# unknown titles or cycles.
from agents.coding_pipeline.architect import _validate_deps

# ---------------------------------------------------------------------------
# Shared imports for decompose structural checks
# ---------------------------------------------------------------------------
from agents.coding_pipeline.models import LeafSpec, TaskTree
from agents.evals.fixtures import EvalFixtureError
from agents.evals.models import GoldCase, GradeCheck, GradeResult
from agents.shared.llm import extract_json

# ---------------------------------------------------------------------------
# Decompose grader
# ---------------------------------------------------------------------------

_PATH_TOKEN_RE = re.compile(r"[A-Za-z0-9_./-]+\.[a-z0-9]{1,5}")


def grade_decompose(case: GoldCase, raw: str) -> GradeResult:
    """Grade a single decompose raw output against *case*.

    Structural checks (always applied)
    -----------------------------------
    1. valid-tree       – extract_json + TaskTree.model_validate, >= 1 leaf.
    2. deps-resolve     – reuse architect._validate_deps (unknown/cycles fail).
    3. files-named      – every leaf's content has >= 1 path-like token or a
                         ``Files`` section unless complexity=novel + Manual.
    4. estimates-bounded – every leaf estimate in {xs,s,m} or novel+Manual.
    5. auto-floors      – every non-Manual leaf: requires_tests=true and
                         (max_files null or >= 3).

    Rubric DSL (from ``expected.rubric``)
    -------------------------------------
    Each rubric item: {id, kind, ...} maps to a GradeCheck named by its ``id``.
    Kinds:

    * ``require_leaf {title_pattern?, content_pattern?}``  – >= 1 leaf matches
      all given regexes.
    * ``forbid_leaf {title_pattern?, content_pattern?}``  – no leaf matches.
    * ``max_leaves {n}`` – total leaf count <= n.
    * ``min_leaves {n}`` – total leaf count >= n.
    * ``require_dep {from_pattern, to_pattern}`` – some matching edge exists.
    * ``require_manual {title_pattern}`` – all matching leaves are Manual.

    Unknown kind -> EvalFixtureError (fail loudly).
    """
    expected: dict[str, Any] = case.expected or {}
    checks: list[GradeCheck] = []

    # -- Structural check 1: valid-tree --------------------------------------
    check_valid_tree = _check_valid_tree(raw)
    checks.append(check_valid_tree)

    tree: TaskTree | None = None
    if check_valid_tree.passed:
        tree = check_valid_tree._tree  # type: ignore[attr-defined]

    if not tree:
        return _result(case.case_id, case.step, checks)

    leaves = tree.leaves

    # -- Structural check 2: deps-resolve ------------------------------------
    checks.append(_check_deps_resolve(leaves))

    # -- Structural check 3: files-named -------------------------------------
    checks.append(_check_files_named(leaves))

    # -- Structural check 4: estimates-bounded -------------------------------
    checks.append(_check_estimates_bounded(leaves))

    # -- Structural check 5: auto-floors -------------------------------------
    checks.append(_check_auto_floors(leaves))

    # -- Rubric checks -------------------------------------------------------
    rubric: list[dict[str, Any]] = expected.get("rubric") or []
    for item in rubric:
        kind = item.get("kind")
        if kind not in _RUBRIC_KINDS:
            raise EvalFixtureError(
                f"unknown rubric kind '{kind}' in case {case.case_id}",
                case.case_dir,
            )
        checks.append(_apply_rubric_kind(kind, item, leaves))

    return _result(case.case_id, case.step, checks)


def _check_valid_tree(raw: str) -> GradeCheck:
    """Structural check 1: JSON parses and yields >= 1 leaf."""
    try:
        data = extract_json(raw)
        if not data:
            return GradeCheck(
                name="valid-tree",
                passed=False,
                detail="raw output is not valid JSON",
            )
        tree = TaskTree.model_validate(data)
        if not tree.leaves:
            return GradeCheck(
                name="valid-tree",
                passed=False,
                detail="task tree has no leaves",
            )
        check = GradeCheck(name="valid-tree", passed=True)
        check._tree = tree  # type: ignore[attr-defined]
        return check
    except ValidationError as exc:
        return GradeCheck(
            name="valid-tree",
            passed=False,
            detail=f"schema validation failed: {exc}",
        )


def _check_deps_resolve(leaves: list[LeafSpec]) -> GradeCheck:
    """Structural check 2: no unknown titles, no cycles."""
    try:
        _validate_deps(leaves)
        return GradeCheck(name="deps-resolve", passed=True)
    except Exception as exc:
        return GradeCheck(
            name="deps-resolve",
            passed=False,
            detail=str(exc),
        )


def _check_files_named(leaves: list[LeafSpec]) -> GradeCheck:
    """Structural check 3: every leaf's content names a file or has Files
    section, unless complexity=novel and execution_mode=Manual."""
    violations: list[str] = []
    for leaf in leaves:
        if leaf.complexity == "novel" and leaf.execution_mode == "Manual":
            continue
        content = leaf.content or ""
        has_path_token = bool(_PATH_TOKEN_RE.search(content))
        has_files_section = bool(re.search(r"(?mi)^files\s*[:=]", content))
        if not has_path_token and not has_files_section:
            violations.append(f"leaf '{leaf.title}' has no file paths")
    if violations:
        return GradeCheck(
            name="files-named",
            passed=False,
            detail="; ".join(violations),
        )
    return GradeCheck(name="files-named", passed=True)


def _check_estimates_bounded(leaves: list[LeafSpec]) -> GradeCheck:
    """Structural check 4: every estimate in {xs,s,m} or novel+Manual."""
    allowed = {"xs", "s", "m", None}
    violations: list[str] = []
    for leaf in leaves:
        est = leaf.estimate
        is_novel = leaf.complexity == "novel"
        is_manual = leaf.execution_mode == "Manual"
        if est is None:
            continue
        if is_novel and is_manual:
            continue
        if est not in allowed:
            violations.append(f"leaf '{leaf.title}' estimate '{est}' not in {{xs,s,m}}")
    if violations:
        return GradeCheck(
            name="estimates-bounded",
            passed=False,
            detail="; ".join(violations),
        )
    return GradeCheck(name="estimates-bounded", passed=True)


def _check_auto_floors(leaves: list[LeafSpec]) -> GradeCheck:
    """Structural check 5: non-Manual leaves need requires_tests=true and
    (max_files null or >= 3)."""
    violations: list[str] = []
    for leaf in leaves:
        if leaf.execution_mode == "Manual":
            continue
        if leaf.requires_tests is not True:
            violations.append(
                f"leaf '{leaf.title}' execution_mode={leaf.execution_mode} "
                f"but requires_tests={leaf.requires_tests}"
            )
        mf = leaf.max_files
        if mf is not None and mf < 3:
            violations.append(f"leaf '{leaf.title}' max_files={mf} < 3")
    if violations:
        return GradeCheck(
            name="auto-floors",
            passed=False,
            detail="; ".join(violations),
        )
    return GradeCheck(name="auto-floors", passed=True)


# ---------------------------------------------------------------------------
# Rubric DSL
# ---------------------------------------------------------------------------

_RUBRIC_KINDS: set[str] = {
    "require_leaf",
    "forbid_leaf",
    "max_leaves",
    "min_leaves",
    "require_dep",
    "require_manual",
}


def _apply_rubric_kind(
    kind: str,
    item: dict[str, Any],
    leaves: list[LeafSpec],
) -> GradeCheck:
    """Dispatch a single rubric item to the correct checker and wrap in GradeCheck."""
    checker = _RUBRIC_CHECKERS[kind]
    ok = checker(kind, item, leaves)
    name = item.get("id", kind)
    return GradeCheck(name=name, passed=ok)


def _match_title(title: str, pattern: str | None) -> bool:
    if pattern is None:
        return True
    return bool(re.search(pattern, title))


def _match_content(content: str | None, pattern: str | None) -> bool:
    if pattern is None:
        return True
    if content is None:
        return False
    return bool(re.search(pattern, content or ""))


def _rubric_check(
    kind: str,
    item: dict[str, Any],
    leaves: list[LeafSpec],
    check_fn,
) -> GradeCheck:
    """Wrap a check function in a GradeCheck named by the rubric item's id."""
    name = item.get("id", kind)
    ok = check_fn(kind, item, leaves)
    return GradeCheck(name=name, passed=ok)


def _check_require_leaf(kind: str, item: dict[str, Any], leaves: list[LeafSpec]) -> bool:
    """At least one leaf matches all given patterns."""
    title_pat = item.get("title_pattern")
    content_pat = item.get("content_pattern")
    for leaf in leaves:
        if _match_title(leaf.title, title_pat) and _match_content(leaf.content, content_pat):
            return True
    return False


def _check_forbid_leaf(kind: str, item: dict[str, Any], leaves: list[LeafSpec]) -> bool:
    """No leaf matches all given patterns."""
    title_pat = item.get("title_pattern")
    content_pat = item.get("content_pattern")
    for leaf in leaves:
        if _match_title(leaf.title, title_pat) and _match_content(leaf.content, content_pat):
            return False
    return True


def _check_max_leaves(kind: str, item: dict[str, Any], leaves: list[LeafSpec]) -> bool:
    """Total leaf count <= n."""
    n = item.get("n", 0)
    return len(leaves) <= n


def _check_min_leaves(kind: str, item: dict[str, Any], leaves: list[LeafSpec]) -> bool:
    """Total leaf count >= n."""
    n = item.get("n", 0)
    return len(leaves) >= n


def _check_require_dep(kind: str, item: dict[str, Any], leaves: list[LeafSpec]) -> bool:
    """At least one edge from a leaf matching from_pattern to a leaf matching
    to_pattern exists."""
    from_pat = item.get("from_pattern")
    to_pat = item.get("to_pattern")
    titles = {leaf.title for leaf in leaves}
    for leaf in leaves:
        if not _match_title(leaf.title, from_pat):
            continue
        for dep in leaf.depends_on:
            if dep in titles and _match_title(dep, to_pat):
                return True
    return False


def _check_require_manual(kind: str, item: dict[str, Any], leaves: list[LeafSpec]) -> bool:
    """All leaves matching the title pattern are Manual."""
    title_pat = item.get("title_pattern")
    for leaf in leaves:
        if _match_title(leaf.title, title_pat) and leaf.execution_mode != "Manual":
            return False
    return True


_RUBRIC_CHECKERS: dict[str, Any] = {
    "require_leaf": _check_require_leaf,
    "forbid_leaf": _check_forbid_leaf,
    "max_leaves": _check_max_leaves,
    "min_leaves": _check_min_leaves,
    "require_dep": _check_require_dep,
    "require_manual": _check_require_manual,
}


# ---------------------------------------------------------------------------
# Boundedness grader
# ---------------------------------------------------------------------------

from agents.coding_pipeline.architect import LeafBoundedness  # noqa: E402


def grade_boundedness(case: GoldCase, raw: str) -> GradeResult:
    """Grade a single boundedness raw output against *case*.

    Parse ``LeafBoundedness``; compare ``worker_shaped`` to expected.
    When ``expected.criteria`` is present, also compare each named criterion
    field. Score = fraction correct; passed = overall verdict correct.
    """
    expected: dict[str, Any] = case.expected or {}
    expected_worker_shaped: bool = expected.get("worker_shaped", False)
    expected_criteria: dict[str, Any] | None = expected.get("criteria")

    checks: list[GradeCheck] = []

    # -- Parse envelope ------------------------------------------------------
    check_parsed = _check_boundedness_parse(raw)
    checks.append(check_parsed)

    if not check_parsed.passed:
        return _result(case.case_id, case.step, checks)

    actual: LeafBoundedness = check_parsed._boundedness  # type: ignore[attr-defined]

    # -- worker_shaped comparison --------------------------------------------
    ws_correct = actual.worker_shaped == expected_worker_shaped
    checks.append(
        GradeCheck(
            name="worker_shaped",
            passed=ws_correct,
            detail=(f"expected={expected_worker_shaped}, actual={actual.worker_shaped}"),
        )
    )

    # -- criteria comparisons ------------------------------------------------
    if expected_criteria:
        for field_name, expected_value in expected_criteria.items():
            actual_value = getattr(actual, field_name, None)
            field_correct = actual_value == expected_value
            checks.append(
                GradeCheck(
                    name=f"criteria.{field_name}",
                    passed=field_correct,
                    detail=(f"expected={expected_value}, actual={actual_value}"),
                )
            )

    return _result(case.case_id, case.step, checks)


def _check_boundedness_parse(raw: str) -> GradeCheck:
    """Check that the raw output parses as a valid LeafBoundedness."""
    try:
        data = extract_json(raw)
        if not data:
            return GradeCheck(
                name="boundedness-parse",
                passed=False,
                detail="raw output is not valid JSON",
            )
        boundedness = LeafBoundedness.model_validate(data)
        check = GradeCheck(name="boundedness-parse", passed=True)
        check._boundedness = boundedness  # type: ignore[attr-defined]
        return check
    except ValidationError as exc:
        return GradeCheck(
            name="boundedness-parse",
            passed=False,
            detail=f"schema validation failed: {exc}",
        )


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


def _result(case_id: str, step: str, checks: list[GradeCheck]) -> GradeResult:
    """Build a GradeResult from a list of checks."""
    clean_checks: list[GradeCheck] = []
    for c in checks:
        clean_checks.append(GradeCheck(name=c.name, passed=c.passed, detail=c.detail))

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
    score = passed_count / total
    all_passed = passed_count == total

    return GradeResult(
        case_id=case_id,
        step=step,  # type: ignore[arg-type]
        passed=all_passed,
        score=round(score, 4),
        checks=clean_checks,
    )
