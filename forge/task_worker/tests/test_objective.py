"""Objective-gate tests — declaration parsing, marker-removal-only diffs, and the
fail-closed paths that stop a dirty-but-irrelevant diff from landing as Done."""

from __future__ import annotations

from types import SimpleNamespace

from forge.task_worker import objective

POINTER_SPEC = (
    "## Spec\n\n"
    "**The spec for this task is the test file: `tests/test_area.py`.** "
    "Objective: make every test in that file pass.\n"
)
LINE_SPEC = "## Spec\n\nSpec-Test: tests/test_storage.py\n"
PLAIN_SPEC = "## Spec\n\nImplement the storage domain.\n"

MARKER_ONLY_DIFF = """\
diff --git a/tests/test_area.py b/tests/test_area.py
--- a/tests/test_area.py
+++ b/tests/test_area.py
@@ -8,12 +8,6 @@
 \"\"\"

 import pytest
-
-pytestmark = pytest.mark.xfail(
-    reason="spec: pipeline:test-as-spec:area-domain not implemented yet", strict=False
-)


 def test_m2_identity():
"""

WEAKENED_DIFF = """\
diff --git a/tests/test_area.py b/tests/test_area.py
--- a/tests/test_area.py
+++ b/tests/test_area.py
@@ -20,7 +20,7 @@
-    assert area.convert(1.0, "m2", "cm2") == pytest.approx(10000.0)
+    assert area.convert(1.0, "m2", "cm2") is not None
"""

CLEAN_SPEC_FILE = '''"""SPEC TEST — mentions the word xfail only in prose."""

def test_ok():
    assert True
'''

MARKED_SPEC_FILE = '''"""SPEC TEST"""
import pytest

pytestmark = pytest.mark.xfail(reason="not implemented yet", strict=False)


def test_ok():
    assert True
'''


class FakeSandbox:
    def __init__(self, returncode: int = 0, stdout: str = "ok"):
        self.returncode = returncode
        self.stdout = stdout
        self.commands: list[list[str]] = []

    def run(self, cmd, *, timeout):
        self.commands.append(cmd)
        return SimpleNamespace(returncode=self.returncode, stdout=self.stdout, stderr="")


def _pytest_repo(tmp_path, spec_rel="tests/test_area.py", spec_content=CLEAN_SPEC_FILE):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\n\n[dependency-groups]\ndev = ["pytest>=8.0"]\n'
        "\n[tool.pytest.ini_options]\ntestpaths = ['tests']\n"
    )
    spec = tmp_path / spec_rel
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(spec_content)
    return tmp_path


def test_spec_test_path_pointer_and_line_and_none():
    assert objective.spec_test_path(POINTER_SPEC) == "tests/test_area.py"
    assert objective.spec_test_path(LINE_SPEC) == "tests/test_storage.py"
    assert objective.spec_test_path(PLAIN_SPEC) is None


def test_marker_removal_only_accepts_marker_diff():
    ok, detail = objective._marker_removal_only(MARKER_ONLY_DIFF)
    assert ok, detail


def test_marker_removal_only_rejects_weakened_assertion():
    ok, detail = objective._marker_removal_only(WEAKENED_DIFF)
    assert not ok
    assert "removed non-marker line" in detail or "added line" in detail


def test_no_declaration_is_vacuous(tmp_path):
    met, out = objective.run_objective(tmp_path, PLAIN_SPEC, sandbox=FakeSandbox())
    assert met and "vacuous" in out


def test_missing_spec_file_fails(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    met, out = objective.run_objective(tmp_path, POINTER_SPEC, sandbox=FakeSandbox())
    assert not met and "missing" in out


def test_surviving_marker_fails_before_running(tmp_path, monkeypatch):
    repo = _pytest_repo(tmp_path, spec_content=MARKED_SPEC_FILE)
    sb = FakeSandbox()
    met, out = objective.run_objective(repo, POINTER_SPEC, sandbox=sb)
    assert not met and "still present" in out
    assert sb.commands == []  # no test run wasted on a marked file


def test_weakened_spec_file_fails(tmp_path, monkeypatch):
    repo = _pytest_repo(tmp_path)
    monkeypatch.setattr(
        "forge.task_worker.vcs.get_file_diff", lambda repo_path, file: WEAKENED_DIFF
    )
    met, out = objective.run_objective(repo, POINTER_SPEC, sandbox=FakeSandbox())
    assert not met and "weakened" in out


def test_clean_implementation_passes(tmp_path, monkeypatch):
    repo = _pytest_repo(tmp_path)
    monkeypatch.setattr(
        "forge.task_worker.vcs.get_file_diff", lambda repo_path, file: MARKER_ONLY_DIFF
    )
    sb = FakeSandbox(returncode=0)
    met, out = objective.run_objective(repo, POINTER_SPEC, sandbox=sb)
    assert met and "objective met" in out
    assert sb.commands[0][:4] == ["uv", "run", "pytest", "--runxfail"]
    assert "tests/test_area.py" in sb.commands[0]


def test_runxfail_failure_fails_gate(tmp_path, monkeypatch):
    repo = _pytest_repo(tmp_path)
    monkeypatch.setattr("forge.task_worker.vcs.get_file_diff", lambda repo_path, file: "")
    sb = FakeSandbox(returncode=1, stdout="1 failed")
    met, out = objective.run_objective(repo, POINTER_SPEC, sandbox=sb)
    assert not met and "--runxfail" in out


def test_non_pytest_repo_fails_closed(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')
    spec = tmp_path / "tests" / "test_area.py"
    spec.parent.mkdir(parents=True)
    spec.write_text(CLEAN_SPEC_FILE)
    monkeypatch.setattr("forge.task_worker.vcs.get_file_diff", lambda repo_path, file: "")
    met, out = objective.run_objective(tmp_path, POINTER_SPEC, sandbox=FakeSandbox())
    assert not met and "pytest repos only" in out
