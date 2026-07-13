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


class _ScriptedSandbox:
    """Returns scripted (returncode, stdout, stderr) tuples per call, in order — lets a
    test drive the exact output a killed vs. failing command produces."""

    def __init__(self, results: list[tuple[int, str, str]]):
        self.results = list(results)
        self.commands: list[list[str]] = []

    def run(self, cmd, *, timeout):
        self.commands.append(cmd)
        rc, out, err = self.results.pop(0)
        return SimpleNamespace(returncode=rc, stdout=out, stderr=err)


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


def test_killed_build_is_labeled_and_flagged_as_infra(tmp_path):
    # A `timeout`-SIGKILLed build exits 137 with no diagnostics; the note must say KILLED
    # and point at container health, not read as a compile failure (the Observinator escape).
    sb = _ScriptedSandbox([(137, "", "")])
    ok, output, ran = tester.run_build(_repo(tmp_path, go=True), sandbox=sb)
    assert (ok, ran) == (False, True)
    assert "KILLED" in output
    assert "container" in output.lower()


def test_timed_out_build_exit_124_flagged_killed(tmp_path):
    sb = _ScriptedSandbox([(124, "", "")])
    ok, output, ran = tester.run_build(_repo(tmp_path, go=True), sandbox=sb)
    assert (ok, ran) == (False, True)
    assert "exit 124" in output and "KILLED" in output


def test_go_test_failure_surfaces_failing_test_name_from_mid_output(tmp_path):
    # `go test ./...` prints the failing package's `--- FAIL: Name` early, then streams
    # trailing `ok` lines from slower passing packages; a plain tail keeps the `ok`s and
    # loses the test name (the Observinator test-gate escape). The digest must recover it.
    go_test_out = (
        "--- FAIL: TestAnalyzer_FingerprintOverCollides_KnownBug (0.00s)\n"
        "    stacktrace_test.go:42: expected distinct fingerprints, got collision\n"
        "FAIL\n"
        "github.com/erewhon/observinator/pkg/analysis  0.02s\n"
        + "\n".join(f"ok  github.com/erewhon/observinator/pkg/p{i}  0.01s" for i in range(60))
        + "\nFAIL\n"
    )
    sb = _ScriptedSandbox([(0, "", ""), (1, go_test_out, "")])  # build ok, test fails
    passed, output = tester.run_tests(_repo(tmp_path, go=True), sandbox=sb)
    assert passed is False
    assert "TestAnalyzer_FingerprintOverCollides_KnownBug" in output
    assert "stacktrace_test.go:42" in output


def test_failure_digest_empty_for_compile_error_leaves_tail_intact(tmp_path):
    # A compile error has no test-failure markers -> digest is empty and the tail (which
    # already carries the compiler diagnostics) is used unchanged.
    sb = _ScriptedSandbox([(1, "", "pkg/x.go:3:2: undefined: foo\n")])
    ok, output, ran = tester.run_build(_repo(tmp_path, go=True), sandbox=sb)
    assert (ok, ran) == (False, True)
    assert "undefined: foo" in output
    assert "--- failing ---" not in output


def test_real_compile_error_gets_exit_code_but_not_kill_hint(tmp_path):
    # A genuine build failure has diagnostics and a normal exit code — it must NOT be
    # mislabeled as a kill; the exit code is recorded and the container-health hint absent.
    sb = _ScriptedSandbox([(1, "", "pkg/x.go:3:2: undefined: foo\n")])
    ok, output, ran = tester.run_build(_repo(tmp_path, go=True), sandbox=sb)
    assert (ok, ran) == (False, True)
    assert "undefined: foo" in output
    assert "exit 1" in output
    assert "KILLED" not in output


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
