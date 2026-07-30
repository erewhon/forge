"""Objective gate for code-as-spec leaves — suite-green is not objective-met.

A leaf whose spec IS a committed xfail test file (test-as-spec) can pass every
other gate with any dirty diff: the unimplemented spec tests are green-by-xfail,
so lint and tests judge a change that never touched the objective. Observed
live (test-as-spec pilot, 2026-07-30): a plugin side-effect write dirtied the
workspace and a no-op session landed as Done. This gate proves the objective:

1. The declared spec test file exists and its module-level ``pytestmark``
   xfail escape hatch is gone.
2. The leaf's working-copy diff to the spec file removes ONLY the marker block
   (plus a then-unused ``import pytest``) — any other edit is test-weakening
   and fails with its own message.
3. The spec file's tests pass under ``--runxfail``, so even a marker the
   textual check missed cannot fake a pass.

The spec-test path is declared in the task spec text itself — either the
canonical pointer sentence ("The spec for this task is the test file:
`tests/test_x.py`") or an explicit ``Spec-Test: tests/test_x.py`` line.
No declaration -> vacuous pass: the gate binds only code-as-spec leaves.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.task_worker.sandbox import Sandbox

_OBJECTIVE_TIMEOUT = 300

# Machine-first form wins over the prose pointer when both are present.
_SPEC_TEST_LINE_RE = re.compile(r"(?im)^\s*spec-test:\s*`?([^`\s]+)`?\s*$")
_SPEC_TEST_POINTER_RE = re.compile(r"(?i)the spec for this task is the test file:\s*`([^`]+)`")

# A module-level pytestmark assignment — the escape hatch that must be gone.
# Deliberately NOT matching the word "xfail" alone: spec files mention it in
# their worker-rules docstring.
_PYTESTMARK_RE = re.compile(r"(?m)^pytestmark\s*=")

# Lines the spec-file diff is allowed to REMOVE: the marker block and its
# then-unused import. Anything else removed (or any line added) is weakening.
_ALLOWED_REMOVED = (
    re.compile(r"^import pytest$"),
    re.compile(r"^pytestmark\s*=\s*pytest\.mark\.xfail\("),
    re.compile(r"^\s*reason\s*=.*$"),
    re.compile(r"^\s*strict\s*=\s*(True|False)\s*,?\s*\)?\s*$"),
    re.compile(r"^\s*\)\s*$"),
    re.compile(r"^\s*$"),
)


def spec_test_path(spec_text: str) -> str | None:
    """The declared spec-test path, or None when the leaf declares none."""
    m = _SPEC_TEST_LINE_RE.search(spec_text)
    if m:
        return m.group(1)
    m = _SPEC_TEST_POINTER_RE.search(spec_text)
    if m:
        return m.group(1)
    return None


def _marker_removal_only(diff_text: str) -> tuple[bool, str]:
    """True when a unified diff touches nothing but the xfail marker block."""
    for line in diff_text.splitlines():
        if line.startswith(("---", "+++", "@@", "diff ", "index ")):
            continue
        if line.startswith("+"):
            body = line[1:]
            if body.strip():
                return False, f"added line: {body.strip()!r}"
        elif line.startswith("-"):
            body = line[1:]
            if not any(rx.match(body) for rx in _ALLOWED_REMOVED):
                return False, f"removed non-marker line: {body.strip()!r}"
    return True, ""


def run_objective(
    repo_path: Path,
    spec_text: str,
    sandbox: Sandbox | None = None,
) -> tuple[bool, str]:
    """Verify the leaf's declared objective. Returns (met, evidence).

    Vacuous pass when the spec declares no spec-test file. Fails closed on a
    missing/renamed spec file, a surviving marker, a weakened spec file, a
    non-pytest repo, or spec tests that do not pass under ``--runxfail``.
    """
    rel = spec_test_path(spec_text)
    if rel is None:
        return True, "no spec-test declared (objective gate vacuous)"

    spec_file = repo_path / rel
    if not spec_file.is_file():
        return False, f"spec test file {rel} is missing — it must not be deleted or renamed"

    try:
        content = spec_file.read_text(encoding="utf-8", errors="ignore")
    except OSError as e:
        return False, f"spec test file {rel} unreadable: {e}"
    if _PYTESTMARK_RE.search(content):
        return False, (
            f"the module-level pytestmark xfail marker is still present in {rel} — "
            "remove it and implement until those tests genuinely pass"
        )

    from forge.task_worker.vcs import VCSError, get_file_diff

    try:
        diff = get_file_diff(repo_path, rel)
    except VCSError as e:
        return False, f"could not diff spec test file {rel}: {e}"
    if diff.strip():
        ok, detail = _marker_removal_only(diff)
        if not ok:
            return False, (
                f"spec test file {rel} was modified beyond removing the xfail marker "
                f"({detail}) — spec tests must not be weakened, changed, or extended"
            )

    from forge.task_worker.tester import _pyproject_has_pytest

    if not _pyproject_has_pytest(repo_path):
        return False, (
            f"spec-test objective declared ({rel}) but the repo has no pytest configuration — "
            "the objective gate currently supports pytest repos only"
        )

    if sandbox is None:
        from forge.task_worker.sandbox import make_sandbox

        sandbox = make_sandbox(repo_path)
    try:
        result = sandbox.run(
            ["uv", "run", "pytest", "--runxfail", "-q", rel], timeout=_OBJECTIVE_TIMEOUT
        )
    except Exception as e:  # noqa: BLE001
        return False, f"objective test run failed to execute: {e}"
    out = (result.stdout or "") + "\n" + (result.stderr or "")
    if result.returncode != 0:
        return False, f"spec tests in {rel} do not pass under --runxfail:\n{out[-1500:]}"
    return True, f"objective met: {rel} passes under --runxfail"
