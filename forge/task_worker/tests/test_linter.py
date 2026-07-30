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


B007 = "src/cli.py:28:29: B007 Loop control variable domain_name not used within loop body"
E501_NEW = "src/cli.py:99:101: E501 Line too long (120 > 100)"


def test_pre_existing_violation_passes_with_disclosure(tmp_path, monkeypatch):
    """Inherited same-file debt must not fail the leaf (the pilot's cli.py violation
    failed every touching leaf for four runs) — pass, with the ignored count stated."""
    sb = FakeSandbox([1, 0, 0, 0, 1, 0], outputs={4: B007 + "\nFound 1 error.\n"})
    monkeypatch.setattr(
        linter,
        "_baseline_keys",
        lambda *a, **k: linter._violation_keys(B007),
    )
    ok, out, fixed = linter.run_lint(_repo(tmp_path, DEP_PYPROJECT), ["src/cli.py"], sandbox=sb)
    assert ok and fixed
    assert "pre-existing" in out and "1" in out


def test_new_violation_fails_and_evidence_names_only_it(tmp_path, monkeypatch):
    sb = FakeSandbox(
        [1, 0, 0, 0, 1, 0], outputs={4: B007 + "\n" + E501_NEW + "\nFound 2 errors.\n"}
    )
    monkeypatch.setattr(
        linter,
        "_baseline_keys",
        lambda *a, **k: linter._violation_keys(B007),
    )
    ok, out, fixed = linter.run_lint(_repo(tmp_path, DEP_PYPROJECT), ["src/cli.py"], sandbox=sb)
    assert not ok and fixed
    assert "E501" in out
    assert "B007" not in out.splitlines()[0]  # the inherited one is not the headline
    assert "pre-existing" in out  # ...but it is disclosed


def test_no_baseline_fails_closed(tmp_path, monkeypatch):
    """When the base revision cannot be established, fail on the full set — never
    guess a violation is inherited."""
    sb = FakeSandbox([1, 0, 0, 0, 1, 0], outputs={4: B007 + "\nFound 1 error.\n"})
    monkeypatch.setattr(linter, "_baseline_keys", lambda *a, **k: None)
    ok, out, fixed = linter.run_lint(_repo(tmp_path, DEP_PYPROJECT), ["src/cli.py"], sandbox=sb)
    assert not ok and "B007" in out


def test_baseline_keys_materializes_and_strips_prefix(tmp_path, monkeypatch):
    """_baseline_keys writes base file copies under .task_worker/, lints them, keys
    the violations by the ORIGINAL path, and cleans up the mirror."""
    from forge.task_worker import vcs as vcs_mod

    (tmp_path / "pyproject.toml").write_text(DEP_PYPROJECT)
    monkeypatch.setattr(vcs_mod, "get_base_file_content", lambda repo, f: "x = 1\n")

    class EchoSandbox:
        def run(self, cmd, *, timeout):
            paths = [c for c in cmd if c.endswith(".py")]
            out = "\n".join(f"{p}:1:1: E501 Line too long (120 > 100)" for p in paths)
            return SimpleNamespace(returncode=1, stdout=out, stderr="")

    keys = linter._baseline_keys(tmp_path, ["src/cli.py"], ["uv", "run", "ruff"], EchoSandbox())
    assert keys == {("src/cli.py", "E501", "Line too long (# > #)")}
    assert not list((tmp_path / ".task_worker").glob("lint_base_*"))  # mirror cleaned up


def test_violation_keys_normalize_digits_and_lines():
    a = linter._violation_keys("src/x.py:10:101: E501 Line too long (105 > 100)")
    b = linter._violation_keys("src/x.py:99:101: E501 Line too long (131 > 100)")
    assert a == b
