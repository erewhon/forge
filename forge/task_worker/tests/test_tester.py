"""Tester-gate tests — sandbox faked; Go detection, compile gate, and the
Go-repo-with-package.json regression under test."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from forge.task_worker import tester


class FakeSandbox:
    """Records commands; returns scripted returncodes per invocation order."""

    def __init__(self, returncodes: list[int]):
        self.returncodes = list(returncodes)
        self.commands: list[list[str]] = []

    def run(self, cmd, *, timeout):
        self.commands.append(cmd)
        rc = self.returncodes.pop(0) if self.returncodes else 0
        return SimpleNamespace(returncode=rc, stdout=f"rc={rc}", stderr="")


def _repo(tmp_path: Path, *, go: bool = False, pkg_scripts: dict | None = None) -> Path:
    if go:
        (tmp_path / "go.mod").write_text("module example.com/x\n\ngo 1.24\n")
    if pkg_scripts is not None:
        (tmp_path / "package.json").write_text(json.dumps({"scripts": pkg_scripts}))
    return tmp_path


def test_go_repo_builds_then_tests(tmp_path):
    sb = FakeSandbox([0, 0])
    passed, _ = tester.run_tests(_repo(tmp_path, go=True), sandbox=sb)
    assert passed is True
    assert sb.commands == [["go", "build", "./..."], ["go", "test", "./..."]]


def test_go_build_failure_short_circuits_before_test(tmp_path):
    # First command (build) fails -> test is never run, gate fails closed.
    sb = FakeSandbox([1, 0])
    passed, _ = tester.run_tests(_repo(tmp_path, go=True), sandbox=sb)
    assert passed is False
    assert sb.commands == [["go", "build", "./..."]]


def test_go_test_failure_fails_gate(tmp_path):
    sb = FakeSandbox([0, 1])  # build ok, tests fail
    passed, _ = tester.run_tests(_repo(tmp_path, go=True), sandbox=sb)
    assert passed is False
    assert sb.commands == [["go", "build", "./..."], ["go", "test", "./..."]]


def test_go_repo_with_package_json_without_test_script_still_uses_go(tmp_path):
    # The exact observinator regression: a Go repo that also carries a
    # package.json with no "test" script must NOT fall through to
    # "no tests configured" — it must compile and test the Go code.
    repo = _repo(tmp_path, go=True, pkg_scripts={"build-web": "vite build"})
    sb = FakeSandbox([0, 0])
    passed, _ = tester.run_tests(repo, sandbox=sb)
    assert passed is True
    assert sb.commands == [["go", "build", "./..."], ["go", "test", "./..."]]


def test_no_tooling_passes_but_discloses_every_probe(tmp_path):
    # A pass without a runner must be a disclosed decision: the reason string
    # names what was probed so "nothing configured" is auditable, not silent.
    passed, output = tester.run_tests(tmp_path, sandbox=FakeSandbox([]))
    assert passed is True
    assert "no test runner configured" in output
    for probe in ("justfile", "go.mod", "pytest", "scripts.test", "Cargo.toml"):
        assert probe in output


def test_run_build_go_repo_compiles(tmp_path):
    sb = FakeSandbox([0])
    ok, _, ran = tester.run_build(_repo(tmp_path, go=True), sandbox=sb)
    assert ok is True
    assert ran is True
    assert sb.commands == [["go", "build", "./..."]]


def test_run_build_go_failure_blocks(tmp_path):
    sb = FakeSandbox([1])
    ok, _, ran = tester.run_build(_repo(tmp_path, go=True), sandbox=sb)
    assert ok is False
    assert ran is True


def test_run_build_noop_for_non_compiled_project(tmp_path):
    # Python/JS repos have no cheap standalone compile step -> ran is False,
    # so the compile gate is a no-op and defers to the test gate.
    sb = FakeSandbox([])
    ok, _, ran = tester.run_build(_repo(tmp_path, pkg_scripts={"test": "vitest"}), sandbox=sb)
    assert ran is False
    assert ok is True
    assert sb.commands == []


# --- static-check gate (run_build) across languages --------------------------------------


def test_run_build_tsconfig_type_checks(tmp_path):
    (tmp_path / "tsconfig.json").write_text("{}")
    sb = FakeSandbox([0])
    ok, output, ran = tester.run_build(tmp_path, sandbox=sb)
    assert (ok, ran) == (True, True)
    assert sb.commands == [["pnpm", "exec", "tsc", "--noEmit"]]
    assert "static-check: typescript" in output


def test_run_build_tsc_failure_blocks(tmp_path):
    # Covers both a real type error and a missing tsc binary — either way the
    # sandbox command exits non-zero and the gate fails closed.
    (tmp_path / "tsconfig.json").write_text("{}")
    ok, _, ran = tester.run_build(tmp_path, sandbox=FakeSandbox([1]))
    assert (ok, ran) == (False, True)


def test_run_build_is_additive_across_languages(tmp_path):
    # Observinator shape: Go backend + TS web assets. Both checks must run —
    # first-match detection would silently skip one of them.
    (tmp_path / "go.mod").write_text("module example.com/x\n")
    (tmp_path / "tsconfig.json").write_text("{}")
    sb = FakeSandbox([0, 0])
    ok, output, ran = tester.run_build(tmp_path, sandbox=sb)
    assert (ok, ran) == (True, True)
    assert sb.commands == [["go", "build", "./..."], ["pnpm", "exec", "tsc", "--noEmit"]]
    assert "static-check: go" in output and "static-check: typescript" in output


def test_run_build_shellchecks_changed_sh_only(tmp_path):
    (tmp_path / "deploy.sh").write_text("#!/bin/sh\n")
    (tmp_path / "other.sh").write_text("#!/bin/sh\n")
    sb = FakeSandbox([0])
    ok, _, ran = tester.run_build(tmp_path, sandbox=sb, changed_files=["deploy.sh"])
    assert (ok, ran) == (True, True)
    assert sb.commands == [["shellcheck", "deploy.sh"]]


def test_run_build_skips_deleted_changed_files(tmp_path):
    # A deletion shows up in the change list but has nothing to check.
    sb = FakeSandbox([])
    ok, _, ran = tester.run_build(tmp_path, sandbox=sb, changed_files=["gone.sh", "gone.py"])
    assert ran is False
    assert sb.commands == []


def test_run_build_py_compiles_changed_python(tmp_path):
    (tmp_path / "mod.py").write_text("x = 1\n")
    sb = FakeSandbox([0])
    ok, _, ran = tester.run_build(tmp_path, sandbox=sb, changed_files=["mod.py"])
    assert (ok, ran) == (True, True)
    assert sb.commands == [["python3", "-m", "py_compile", "mod.py"]]


def test_run_build_shellcheck_failure_blocks(tmp_path):
    (tmp_path / "deploy.sh").write_text("#!/bin/sh\n")
    ok, _, ran = tester.run_build(tmp_path, sandbox=FakeSandbox([1]), changed_files=["deploy.sh"])
    assert (ok, ran) == (False, True)


# --- Python syntax floor in the test gate -------------------------------------------------


def test_python_repo_without_pytest_gets_syntax_floor(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')
    (tmp_path / "mod.py").write_text("x = 1\n")
    sb = FakeSandbox([0])
    passed, _ = tester.run_tests(tmp_path, sandbox=sb, changed_files=["mod.py"])
    assert passed is True
    assert sb.commands == [["python3", "-m", "py_compile", "mod.py"]]


def test_python_syntax_floor_without_change_list_compiles_tree(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')
    sb = FakeSandbox([0])
    passed, _ = tester.run_tests(tmp_path, sandbox=sb)
    assert passed is True
    assert sb.commands[0][:3] == ["python3", "-m", "compileall"]


def test_python_repo_with_pytest_still_runs_pytest(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    sb = FakeSandbox([0])
    passed, _ = tester.run_tests(tmp_path, sandbox=sb, changed_files=["mod.py"])
    assert passed is True
    assert sb.commands == [["uv", "run", "pytest"]]
