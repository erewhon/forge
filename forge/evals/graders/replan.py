"""Deterministic grader for the `replan` step.

Judges the model's RAW output -- validity is part of the grade.
Mirror ``panel._parse_model`` semantics: extract JSON then validate
against ``ReplanEnvelope``; on any parse/validation failure the
envelope check fails and scoring short-circuits.

Contract
--------
``grade(case, raw) -> GradeResult``

Expected block keys in ``case.expected``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* ``must`` -- list of dicts with ``kind`` and optional target fields
  (``finding_slug``, ``leaf_title``, ``feature``).  An action matches a
  must entry when its ``kind`` equals and every given target field is
  present and equal.
* ``forbid_kinds`` -- action ``kind`` values that must not appear.
* ``forbid_targets`` -- leaf titles the model must not touch
  (the escalated list).
* ``allow_extra`` -- when ``false`` (default), no unmatched actions are
  allowed.  When ``true``, extra actions are skip-passed.
* ``require_empty`` -- when ``true``, the actions list must be empty.

Checks
^^^^^^
1. ``valid-envelope`` -- parses + schema-validates.
2. ``must-actions`` -- every ``must`` entry matched by >= 1 action.
3. ``no-forbidden`` -- no action has a forbidden kind or targets.
4. ``fixup-confirmed-only`` -- every fixup's ``finding_slug`` is a
   CONFIRMED finding in the case's ``report.json``.
5. ``no-extras`` -- extra / require_empty enforcement.
6. ``leaf-floors`` -- new/revised LeafSpec validation.

Scoring
^^^^^^^
``score`` = passed_checks / total_applicable_checks
``passed``  = all applicable checks pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

# ---------------------------------------------------------------------------
# ReplanEnvelope / ReplanAction import
# ---------------------------------------------------------------------------
from agents.coding_pipeline.architect import ReplanEnvelope
from agents.coding_pipeline.models import (
    FixupAction,
    IntegrationFixAction,
    ReplanAction,
    RespecAction,
)
from agents.evals.models import GoldCase, GradeCheck, GradeResult
from agents.shared.llm import extract_json


def grade(case: GoldCase, raw: str) -> GradeResult:
    """Grade a single replan raw output against *case*."""
    expected: dict[str, Any] = case.expected or {}
    must: list[dict[str, Any]] = expected.get("must") or []
    forbid_kinds: list[str] = expected.get("forbid_kinds") or []
    forbid_targets: list[str] = expected.get("forbid_targets") or []
    allow_extra: bool = bool(expected.get("allow_extra", False))
    require_empty: bool = bool(expected.get("require_empty", False))

    checks: list[GradeCheck] = []

    # -- Step 1: parse envelope -----------------------------------------------
    check_valid = _check_valid_envelope(raw)
    checks.append(check_valid)

    if not check_valid.passed:
        return _result(case.case_id, case.step, checks)

    envelope: ReplanEnvelope = check_valid._envelope  # type: ignore[attr-defined]
    actions: list[ReplanAction] = envelope.actions

    # -- Step 2: must-actions -------------------------------------------------
    checks.append(_check_must_actions(actions, must))

    # -- Step 3: no-forbidden -------------------------------------------------
    checks.append(_check_no_forbidden(actions, forbid_kinds, forbid_targets))

    # -- Step 4: fixup-confirmed-only -----------------------------------------
    checks.append(_check_fixup_confirmed_only(actions, case.case_dir))

    # -- Step 5: no-extras ----------------------------------------------------
    checks.append(_check_no_extras(actions, must, allow_extra, require_empty))

    # -- Step 6: leaf-floors --------------------------------------------------
    checks.append(_check_leaf_floors(actions))

    return _result(case.case_id, case.step, checks)


# ---------------------------------------------------------------------------
# Check implementations
# ---------------------------------------------------------------------------


def _check_valid_envelope(raw: str) -> GradeCheck:
    """Check 1: parses + schema-validates."""
    try:
        data = extract_json(raw)
        if not data:
            return GradeCheck(
                name="valid-envelope",
                passed=False,
                detail="raw output is not valid JSON",
            )
        env = ReplanEnvelope.model_validate(data)
        check = GradeCheck(name="valid-envelope", passed=True)
        # stash on the check for the caller
        check._envelope = env  # type: ignore[attr-defined]
        return check
    except ValidationError as exc:
        return GradeCheck(
            name="valid-envelope",
            passed=False,
            detail=f"schema validation failed: {exc}",
        )


def _action_kind(action: ReplanAction) -> str:
    return action.kind


def _action_matches(action: ReplanAction, spec: dict[str, Any]) -> bool:
    """Does *action* match a must-entry *spec*?"""
    if _action_kind(action) != spec.get("kind"):
        return False
    if isinstance(action, FixupAction):
        if "finding_slug" in spec and action.finding_slug != spec["finding_slug"]:
            return False
    elif isinstance(action, RespecAction):
        if "leaf_title" in spec and action.leaf_title != spec["leaf_title"]:
            return False
    elif isinstance(action, IntegrationFixAction):
        if "feature" in spec and action.leaf.feature != spec["feature"]:
            return False
    # SplitSubtree, Escalate, Halt: only kind matters (no extra must fields in current fixtures)
    return True


def _check_must_actions(actions: list[ReplanAction], must: list[dict[str, Any]]) -> GradeCheck:
    """Check 2: every must entry matched by >= 1 action."""
    if not must:
        return GradeCheck(name="must-actions", passed=True)
    failures: list[str] = []
    for spec in must:
        if not any(_action_matches(a, spec) for a in actions):
            failures.append(f"kind={spec.get('kind')}")
    if failures:
        return GradeCheck(
            name="must-actions",
            passed=False,
            detail=f"unmatched must entries: {', '.join(failures)}",
        )
    return GradeCheck(name="must-actions", passed=True)


def _check_no_forbidden(
    actions: list[ReplanAction],
    forbid_kinds: list[str],
    forbid_targets: list[str],
) -> GradeCheck:
    """Check 3: no action has a forbidden kind or targets."""
    violations: list[str] = []

    for action in actions:
        kind = _action_kind(action)
        if kind in forbid_kinds:
            violations.append(f"forbidden kind: {kind}")

        # Check target fields against forbid_targets
        if isinstance(action, FixupAction):
            if action.leaf.title in forbid_targets:
                violations.append(f"forbidden target (fixup leaf): {action.leaf.title}")
        elif isinstance(action, RespecAction):
            if action.leaf_title in forbid_targets:
                violations.append(f"forbidden target (respec title): {action.leaf_title}")
            if action.revised.title in forbid_targets:
                violations.append(f"forbidden target (respec revised): {action.revised.title}")
        elif isinstance(action, IntegrationFixAction):
            if action.leaf.title in forbid_targets:
                violations.append(f"forbidden target (intfix leaf): {action.leaf.title}")

    if violations:
        return GradeCheck(
            name="no-forbidden",
            passed=False,
            detail="; ".join(violations),
        )
    return GradeCheck(name="no-forbidden", passed=True)


def _check_fixup_confirmed_only(actions: list[ReplanAction], case_dir: Path) -> GradeCheck:
    """Check 4: every fixup's finding_slug is CONFIRMED in report.json."""
    # Load confirmed finding slugs from the case's report.json
    confirmed_slugs: set[str] = set()

    report_path = case_dir / "report.json"
    if report_path.exists():
        try:
            report_data = json.loads(report_path.read_text())
            # report.json contains a WaveReport with a "findings" list
            # Each finding has a "slug" and "confirmed" boolean
            if isinstance(report_data, dict):
                findings = report_data.get("findings") or report_data.get("findings", [])
                if isinstance(findings, list):
                    for f in findings:
                        if isinstance(f, dict) and f.get("confirmed"):
                            confirmed_slugs.add(f["slug"])
        except (json.JSONDecodeError, OSError):
            pass

    # If no report.json, all fixups pass (nothing to check against)
    if not confirmed_slugs and not report_path.exists():
        return GradeCheck(name="fixup-confirmed-only", passed=True)

    violations: list[str] = []
    for action in actions:
        if _action_kind(action) == "fixup" and isinstance(action, FixupAction):
            if action.finding_slug not in confirmed_slugs:
                violations.append(f"fixup finding_slug '{action.finding_slug}' is not confirmed")

    if violations:
        return GradeCheck(
            name="fixup-confirmed-only",
            passed=False,
            detail="; ".join(violations),
        )
    return GradeCheck(name="fixup-confirmed-only", passed=True)


def _check_no_extras(
    actions: list[ReplanAction],
    must: list[dict[str, Any]],
    allow_extra: bool,
    require_empty: bool,
) -> GradeCheck:
    """Check 5: extra / require_empty enforcement."""
    if require_empty:
        if actions:
            return GradeCheck(
                name="no-extras",
                passed=False,
                detail=f"require_empty is true but {len(actions)} action(s) present",
            )
        return GradeCheck(name="no-extras", passed=True)

    if allow_extra:
        return GradeCheck(name="no-extras", passed=True)

    # Count how many must entries are matched by at least one action
    matched = set()
    for spec in must:
        for i, a in enumerate(actions):
            if _action_matches(a, spec):
                matched.add(i)

    unmatched = len(actions) - len(matched)
    if unmatched > 0:
        return GradeCheck(
            name="no-extras",
            passed=False,
            detail=f"{unmatched} unmatched action(s) with allow_extra=false",
        )
    return GradeCheck(name="no-extras", passed=True)


# Small/medium estimates are the allowed floor for new/revised leaves.
_ESTIMATE_FLOOR = {"xs", "s", "m"}
_MIN_CONTENT_CHARS = 200


def _check_leaf_floors(actions: list[ReplanAction]) -> GradeCheck:
    """Check 6: every new/revised LeafSpec: estimate in {xs,s,m};
    non-Manual leaves have requires_tests=true; content >= 200 chars."""
    violations: list[str] = []
    _seen: set[str] = set()

    for action in actions:
        leaf: Any | None = None
        if _action_kind(action) == "fixup" and isinstance(action, FixupAction):
            leaf = action.leaf
        elif _action_kind(action) == "respec" and isinstance(action, RespecAction):
            leaf = action.revised
        elif _action_kind(action) == "integration_fix" and isinstance(action, IntegrationFixAction):
            leaf = action.leaf

        if leaf is None:
            continue

        # Avoid double-counting when same leaf appears in fixup + respec
        title = leaf.title
        if title in _seen:
            continue
        _seen.add(title)

        # Estimate floor check
        estimate = getattr(leaf, "estimate", None)
        if estimate is not None and estimate not in _ESTIMATE_FLOOR:
            violations.append(f"leaf '{title}' estimate '{estimate}' not in {{xs,s,m}}")

        # requires_tests check (only for non-Manual)
        exec_mode = getattr(leaf, "execution_mode", "Manual")
        if exec_mode != "Manual":
            if getattr(leaf, "requires_tests", True) is not True:
                violations.append(
                    f"leaf '{title}' execution_mode={exec_mode} but requires_tests is not true"
                )

        # Content non-trivial check
        content = getattr(leaf, "content", "")
        if isinstance(content, str) and len(content) < _MIN_CONTENT_CHARS:
            violations.append(
                f"leaf '{title}' content is {len(content)} chars (minimum {_MIN_CONTENT_CHARS})"
            )

    if violations:
        return GradeCheck(
            name="leaf-floors",
            passed=False,
            detail="; ".join(violations),
        )
    return GradeCheck(name="leaf-floors", passed=True)


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


def _result(case_id: str, step: str, checks: list[GradeCheck]) -> GradeResult:
    """Build a GradeResult from a list of checks."""
    # Filter out internal stashed data
    clean_checks = []
    for c in checks:
        clean_checks.append(GradeCheck(name=c.name, passed=c.passed, detail=c.detail))

    applicable = clean_checks  # all checks are applicable
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
