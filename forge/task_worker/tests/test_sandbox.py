"""Sandbox seam tests: protocol conformance, GaolDx delegation, factory, and consumers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agents.task_worker import executor as ex
from agents.task_worker import sandbox as sb
from agents.task_worker import tester
from agents.task_worker.models import TaskInfo
from agents.task_worker.sandbox import GaolDxSandbox, Sandbox, make_sandbox


class FakeSandbox:
    """A protocol-conforming fake: records commands, returns canned results."""

    def __init__(self, repo: Path, *, returncode: int = 0, stdout: str = "ok", raises=None):
        self.repo = repo
        self.returncode = returncode
        self.stdout = stdout
        self.raises = raises
        self.commands: list[list[str]] = []

    def preflight(self) -> tuple[bool, str]:
        return True, "fake ready"

    def run(self, cmd: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
        self.commands.append(cmd)
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(cmd, self.returncode, stdout=self.stdout, stderr="")

    def run_tests(self) -> tuple[bool, str]:
        return tester.run_tests(self.repo, sandbox=self)


def test_fake_satisfies_protocol(tmp_path):
    assert isinstance(FakeSandbox(tmp_path), Sandbox)
    assert isinstance(GaolDxSandbox(tmp_path), Sandbox)


# --- GaolDxSandbox delegation ---------------------------------------------------


def test_gaol_dx_delegates_to_dx_helpers(tmp_path, monkeypatch):
    monkeypatch.setattr(sb, "check_dx_ready", lambda repo: (True, "dx status: running"))
    seen = {}

    def fake_dx_run(repo, cmd, timeout):
        seen.update(repo=repo, cmd=cmd, timeout=timeout)
        return subprocess.CompletedProcess(cmd, 0, stdout="done", stderr="")

    monkeypatch.setattr(sb, "dx_run", fake_dx_run)
    box = GaolDxSandbox(tmp_path)
    assert box.preflight() == (True, "dx status: running")
    result = box.run(["echo", "hi"], timeout=5)
    assert result.stdout == "done"
    assert seen == {"repo": tmp_path, "cmd": ["echo", "hi"], "timeout": 5}


def test_gaol_dx_run_tests_delegates_to_tester(tmp_path, monkeypatch):
    seen = {}

    def fake_run_tests(repo, sandbox=None):
        seen.update(repo=repo, sandbox=sandbox)
        return True, "green"

    monkeypatch.setattr(tester, "run_tests", fake_run_tests)
    box = GaolDxSandbox(tmp_path)
    assert box.run_tests() == (True, "green")
    assert seen["repo"] == tmp_path
    assert seen["sandbox"] is box  # tester runs commands through the same sandbox


# --- factory ---------------------------------------------------------------------


def test_make_sandbox_default_is_gaol_dx(tmp_path):
    assert isinstance(make_sandbox(tmp_path), GaolDxSandbox)


def test_make_sandbox_unknown_kind_raises(tmp_path, monkeypatch):
    from agents.task_worker.config import settings

    monkeypatch.setattr(settings, "sandbox", "warp-drive")
    with pytest.raises(ValueError, match="warp-drive"):
        make_sandbox(tmp_path)


# --- tester through the seam --------------------------------------------------------


def test_tester_detects_pytest_and_runs_via_sandbox(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    box = FakeSandbox(tmp_path)
    passed, out = tester.run_tests(tmp_path, sandbox=box)
    assert passed
    assert box.commands == [["uv", "run", "pytest"]]


def test_tester_timeout_message_preserved(tmp_path):
    (tmp_path / "pyproject.toml").write_text("pytest\n")
    box = FakeSandbox(tmp_path, raises=subprocess.TimeoutExpired(cmd="pytest", timeout=300))
    passed, out = tester.run_tests(tmp_path, sandbox=box)
    assert not passed
    assert "TIMEOUT after 300s" in out


def test_tester_no_config_skips_sandbox_entirely(tmp_path):
    box = FakeSandbox(tmp_path)
    passed, out = tester.run_tests(tmp_path, sandbox=box)
    assert passed and out == "no tests configured"
    assert box.commands == []


# --- executor through the seam -------------------------------------------------------


def _task() -> TaskInfo:
    return TaskInfo(
        id="row-1",
        task="t",
        project="Meta",
        status="Ready",
        priority=2,
        execution_mode="Auto-OK",
    )


def test_executor_runs_opencode_via_sandbox_and_cleans_spec(tmp_path):
    box = FakeSandbox(tmp_path, stdout="did the thing")
    ok, tail, blocked = ex.execute_task_with_opencode(
        _task(), "SPEC", tmp_path, "auto", 60, sandbox=box
    )
    assert ok and not blocked
    assert box.commands and box.commands[0][0] == "opencode"
    assert "llm/auto" in box.commands[0]
    leftover = list((tmp_path / ".task_worker").glob("spec-*.md"))
    assert leftover == []  # spec cleaned up on the way out


def test_executor_blocked_marker_fails_via_sandbox(tmp_path):
    box = FakeSandbox(tmp_path, stdout="BLOCKED: cannot proceed")
    ok, tail, blocked = ex.execute_task_with_opencode(
        _task(), "SPEC", tmp_path, "auto", 60, sandbox=box
    )
    assert not ok and blocked
    assert "BLOCKED" in tail


def test_executor_blocked_marker_detected_through_ansi(tmp_path):
    box = FakeSandbox(tmp_path, stdout="work done\n\x1b[91mBLOCKED:\x1b[0m missing dep\n")
    ok, _, blocked = ex.execute_task_with_opencode(
        _task(), "SPEC", tmp_path, "auto", 60, sandbox=box
    )
    assert not ok and blocked


def test_executor_mid_line_blocked_mention_is_not_a_refusal(tmp_path):
    box = FakeSandbox(tmp_path, stdout="I will print BLOCKED: only if I cannot proceed. Done.")
    ok, _, blocked = ex.execute_task_with_opencode(
        _task(), "SPEC", tmp_path, "auto", 60, sandbox=box
    )
    assert ok and not blocked
