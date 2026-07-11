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


def test_no_tooling_reports_no_tests(tmp_path):
    passed, output = tester.run_tests(tmp_path, sandbox=FakeSandbox([]))
    assert passed is True
    assert "no tests configured" in output


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
