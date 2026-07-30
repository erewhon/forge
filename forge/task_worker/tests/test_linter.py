"""Lint-gate tests — sandbox faked; detection, autofix-then-recheck, and scoping under test."""

from __future__ import annotations

from types import SimpleNamespace

from forge.task_worker import linter


class FakeSandbox:
    """Records commands; returns scripted returncodes per invocation order."""

    def __init__(self, returncodes: list[int], outputs: dict[int, str] | None = None):
        self.returncodes = list(returncodes)
        self.outputs = outputs or {}  # invocation index -> stdout
        self.commands: list[list[str]] = []

    def run(self, cmd, *, timeout):
        idx = len(self.commands)
        self.commands.append(cmd)
        rc = self.returncodes.pop(0) if self.returncodes else 0
        return SimpleNamespace(returncode=rc, stdout=self.outputs.get(idx, f"rc={rc}"), stderr="")


def _repo(tmp_path, pyproject: str | None):
    if pyproject is not None:
        (tmp_path / "pyproject.toml").write_text(pyproject)
    return tmp_path


DEP_PYPROJECT = '[project]\nname = "x"\n\n[dependency-groups]\ndev = ["ruff>=0.4"]\n'
CONFIG_ONLY_PYPROJECT = '[project]\nname = "x"\n\n[tool.ruff]\nline-length = 100\n'


def test_no_python_files_is_vacuous(tmp_path):
    ok, out, fixed = linter.run_lint(
        _repo(tmp_path, DEP_PYPROJECT), ["README.md", "data.json"], sandbox=FakeSandbox([])
    )
    assert ok and not fixed and "no lintable" in out


def test_no_ruff_intent_is_vacuous(tmp_path):
    ok, out, fixed = linter.run_lint(
        _repo(tmp_path, '[project]\nname = "x"\n'), ["a.py"], sandbox=FakeSandbox([])
    )
    assert ok and not fixed and "no linter configured" in out


def test_clean_first_pass_runs_no_autofix(tmp_path):
    sb = FakeSandbox([0, 0])  # check, format --check
    ok, out, fixed = linter.run_lint(_repo(tmp_path, DEP_PYPROJECT), ["a.py", "b.py"], sandbox=sb)
    assert ok and not fixed
    assert len(sb.commands) == 2
    assert sb.commands[0][:3] == ["uv", "run", "ruff"]
    # changed files only — never the whole repo
    assert sb.commands[0][-2:] == ["a.py", "b.py"]
    assert "--fix" not in [arg for cmd in sb.commands for arg in cmd]


def test_violations_autofixed_then_pass(tmp_path):
    # check fails, format ok → fix + format → recheck clean
    sb = FakeSandbox([1, 0, 0, 0, 0, 0])
    ok, out, fixed = linter.run_lint(_repo(tmp_path, DEP_PYPROJECT), ["a.py"], sandbox=sb)
    assert ok and fixed
    flat = [" ".join(cmd) for cmd in sb.commands]
    assert any("check --fix" in c for c in flat)
    assert any(c.endswith("format a.py") for c in flat)


def test_surviving_violations_fail_the_gate(tmp_path):
    # check fails, autofix runs, recheck STILL fails
    sb = FakeSandbox([1, 0, 0, 0, 1, 0])
    ok, out, fixed = linter.run_lint(_repo(tmp_path, DEP_PYPROJECT), ["a.py"], sandbox=sb)
    assert not ok and fixed


def test_config_only_repo_uses_uvx(tmp_path):
    sb = FakeSandbox([0, 0])
    ok, _, _ = linter.run_lint(_repo(tmp_path, CONFIG_ONLY_PYPROJECT), ["a.py"], sandbox=sb)
    assert ok
    assert sb.commands[0][:3] == ["uvx", "-q", "ruff"]


def test_failure_evidence_strips_uv_noise(tmp_path):
    """The failing tail must carry ruff's violations, not uv download chatter —
    the retry prompt is built from it (a cold-cache sandbox produced evidence that
    was 100% download noise, so the model looped on an invisible violation)."""
    noise = "Downloading ruff (10.9MiB)\n Downloaded ruff\nInstalled 1 package in 18ms\n"
    violation = "src/x.py:1:1: E501 Line too long (120 > 100)\nFound 1 error.\n"
    sb = FakeSandbox([1, 0, 0, 0, 1, 0], outputs={4: violation + noise * 40})
    ok, out, fixed = linter.run_lint(_repo(tmp_path, DEP_PYPROJECT), ["a.py"], sandbox=sb)
    assert not ok and fixed
    assert "E501" in out
    assert "Downloading" not in out
