"""Sandbox seam tests: protocol conformance, GaolDx delegation, factory, and consumers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from forge.task_worker import executor as ex
from forge.task_worker import sandbox as sb
from forge.task_worker import tester
from forge.task_worker.models import TaskInfo
from forge.task_worker.sandbox import (
    GaolDxSandbox,
    GaolRunOnceSandbox,
    Sandbox,
    make_sandbox,
)


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
    assert isinstance(GaolRunOnceSandbox(tmp_path), Sandbox)


# --- GaolDxSandbox delegation ---------------------------------------------------


def test_gaol_dx_delegates_to_dx_helpers(tmp_path, monkeypatch):
    monkeypatch.setattr(sb, "check_dx_ready", lambda repo: (True, "dx status: running"))
    monkeypatch.setattr(sb, "check_disk_free", lambda repo, floor: (True, "disk ok: 9000 MiB free"))
    seen = {}

    def fake_dx_run(repo, cmd, timeout):
        seen.update(repo=repo, cmd=cmd, timeout=timeout)
        return subprocess.CompletedProcess(cmd, 0, stdout="done", stderr="")

    monkeypatch.setattr(sb, "dx_run", fake_dx_run)
    box = GaolDxSandbox(tmp_path)
    # preflight composes the container status with the disk-space status
    assert box.preflight() == (True, "dx status: running; disk ok: 9000 MiB free")
    result = box.run(["echo", "hi"], timeout=5)
    assert result.stdout == "done"
    assert seen["repo"] == tmp_path
    # the command is wrapped in a container-side timeout so it can't outlive the call
    # (orphaned in-container processes keep writing into the bind-mounted repo)
    assert seen["cmd"] == ["timeout", "--kill-after=30", "5s", "echo", "hi"]
    # host-side kill is a delayed backstop — it must never fire before the inner timeout
    assert seen["timeout"] == 5 + GaolDxSandbox._HOST_GRACE_S


def test_gaol_dx_preflight_refuses_when_disk_low(tmp_path, monkeypatch):
    # A near-full container must be refused BEFORE work starts, so the leaf is skipped with
    # the real cause named — not run, killed at the gate, and blamed for the empty output.
    monkeypatch.setattr(sb, "check_dx_ready", lambda repo: (True, "dx status: running"))
    monkeypatch.setattr(
        sb,
        "check_disk_free",
        lambda repo, floor: (False, "container disk low: 300 MiB free (< 2048 MiB floor)"),
    )
    ready, status = GaolDxSandbox(tmp_path).preflight()
    assert not ready
    assert "container disk low" in status


def test_gaol_dx_preflight_skips_disk_when_container_not_ready(tmp_path, monkeypatch):
    # If the container isn't up, report that — don't probe disk against a dead container.
    monkeypatch.setattr(sb, "check_dx_ready", lambda repo: (False, "dx status: stopped"))
    called = {"disk": False}

    def _disk(repo, floor):
        called["disk"] = True
        return True, "disk ok"

    monkeypatch.setattr(sb, "check_disk_free", _disk)
    ready, status = GaolDxSandbox(tmp_path).preflight()
    assert (ready, status) == (False, "dx status: stopped")
    assert called["disk"] is False


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


# --- GaolRunOnceSandbox ------------------------------------------------------------


@pytest.fixture
def bare_home(monkeypatch, tmp_path):
    """A fake $HOME with no opencode setup, no extra mounts, and no extra hosts, so the
    host's real config/dirs (or a local .env) never leak into argv assertions."""
    from forge.task_worker.config import settings as tw_settings

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(sb.Path, "home", staticmethod(lambda: home))
    monkeypatch.setattr(tw_settings, "runonce_extra_mounts", [])
    monkeypatch.setattr(tw_settings, "runonce_extra_hosts", [])
    return home


def _capture_run(monkeypatch):
    seen: dict = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(sb.subprocess, "run", fake_run)
    return seen


def test_run_once_argv_shape(tmp_path, monkeypatch, bare_home):
    """The workspace is mounted writable at the SAME path (opencode derives its project
    root from PWD), caps and the in-container timeout are set, host-resolved LAN names
    are injected, the network readiness gate is on, and the command follows the ``--``
    separator verbatim."""
    from forge.task_worker.config import settings as tw_settings

    seen = _capture_run(monkeypatch)
    monkeypatch.setattr(tw_settings, "runonce_extra_hosts", ["llm-router.internal"])
    monkeypatch.setattr(sb.socket, "gethostbyname", lambda name: "192.0.2.7")
    repo = tmp_path / "cw-leaf"
    repo.mkdir()
    box = GaolRunOnceSandbox(repo)
    result = box.run(["uv", "run", "pytest"], timeout=600)
    assert result.stdout == "ok"

    args = seen["args"]
    ws = str(repo)
    assert args[:2] == ["gaol", "run-once"]
    assert args[args.index("--runtime") + 1] == "incus"
    assert args[args.index("--image") + 1] == "gaol-candidate-base"
    assert f"{ws}:{ws}" in args  # same-path writable mount
    assert args[args.index("--workdir") + 1] == ws
    assert f"PWD={ws}" in args
    assert "HOME=/home/dev" in args
    assert args[args.index("--memory") + 1] == "4GiB"
    assert args[args.index("--cpus") + 1] == "2"
    # LAN/mesh names the NAT'd sandbox DNS can't resolve, injected host-resolved
    assert args[args.index("--add-host") + 1] == "llm-router.internal:192.0.2.7"
    # DHCP readiness gate — a network command must not start before the NIC is up
    assert args[args.index("--wait-network") + 1] == "30"
    assert args[args.index("--timeout") + 1] == "600"
    assert args[args.index("--") + 1 :] == ["uv", "run", "pytest"]
    # host-side kill is a delayed backstop beyond the in-container timeout
    assert seen["kwargs"]["timeout"] == 600 + GaolRunOnceSandbox._HOST_GRACE_S
    assert seen["kwargs"]["cwd"] == repo


def test_run_once_extra_mounts_same_path_and_missing_skipped(tmp_path, monkeypatch, bare_home):
    """Out-of-repo path deps and the uv cache ride along same-path; a missing entry is
    skipped (never a crash, never a mount incus would reject as absent)."""
    from forge.task_worker.config import settings as tw_settings

    present = tmp_path / "nous"
    present.mkdir()
    missing = tmp_path / "not-there"
    monkeypatch.setattr(tw_settings, "runonce_extra_mounts", [present, missing])
    seen = _capture_run(monkeypatch)
    GaolRunOnceSandbox(tmp_path / "ws").run(["true"], timeout=5)
    mounts = [seen["args"][i + 1] for i, a in enumerate(seen["args"]) if a == "--mount"]
    assert f"{present}:{present}" in mounts
    assert not any(str(missing) in m for m in mounts)


def test_run_once_no_extra_mounts_by_config(tmp_path, monkeypatch, bare_home):
    from forge.task_worker.config import settings as tw_settings

    monkeypatch.setattr(tw_settings, "runonce_extra_mounts", [])
    seen = _capture_run(monkeypatch)
    GaolRunOnceSandbox(tmp_path).run(["true"], timeout=5)
    mounts = [seen["args"][i + 1] for i, a in enumerate(seen["args"]) if a == "--mount"]
    assert mounts == [f"{tmp_path}:{tmp_path}"]  # just the workspace


def test_run_once_unresolvable_extra_host_is_skipped(tmp_path, monkeypatch, bare_home):
    from forge.task_worker.config import settings as tw_settings

    def boom(name):
        raise OSError("no such host")

    seen = _capture_run(monkeypatch)
    monkeypatch.setattr(tw_settings, "runonce_extra_hosts", ["llm-router.internal"])
    monkeypatch.setattr(sb.socket, "gethostbyname", boom)
    GaolRunOnceSandbox(tmp_path).run(["true"], timeout=5)
    assert "--add-host" not in seen["args"]  # skipped, never a crash


def test_run_once_metachars_pass_through_as_single_elements(tmp_path, monkeypatch, bare_home):
    # gaol re-joins argv into an unquoted shell string (backticks execute!) — the sandbox
    # must never quote, join, or mangle an element on its way through.
    seen = _capture_run(monkeypatch)
    box = GaolRunOnceSandbox(tmp_path)
    box.run(["echo", "`boom` && $(id)"], timeout=5)
    assert seen["args"][-1] == "`boom` && $(id)"
    assert seen["args"][-2] == "echo"


def test_run_once_without_host_opencode_mounts_only_the_workspace(tmp_path, monkeypatch, bare_home):
    seen = _capture_run(monkeypatch)
    box = GaolRunOnceSandbox(tmp_path)
    box.run(["true"], timeout=5)
    mounts = [seen["args"][i + 1] for i, a in enumerate(seen["args"]) if a == "--mount"]
    assert mounts == [f"{tmp_path}:{tmp_path}"]
    assert not (tmp_path / ".task_worker").exists()  # no state dir created for nothing


def test_run_once_mounts_opencode_config_and_seeds_private_state(tmp_path, monkeypatch, bare_home):
    """Host config is shared read-mostly; the data dir is per-sandbox, seeded with ONLY
    auth.json (private opencode.db per worker — concurrent sessions corrupt the shared
    sqlite WAL), and lives under the repo's self-ignored .task_worker/."""
    (bare_home / ".config" / "opencode").mkdir(parents=True)
    auth_dir = bare_home / ".local" / "share" / "opencode"
    auth_dir.mkdir(parents=True)
    (auth_dir / "auth.json").write_text('{"llm": "key"}')
    (auth_dir / "opencode.db").write_text("HOST DB — must not be copied")

    seen = _capture_run(monkeypatch)
    repo = tmp_path / "ws"
    repo.mkdir()
    box = GaolRunOnceSandbox(repo)
    box.run(["true"], timeout=5)

    mounts = [seen["args"][i + 1] for i, a in enumerate(seen["args"]) if a == "--mount"]
    state = repo / ".task_worker" / "opencode-state"
    assert mounts == [
        f"{repo}:{repo}",
        f"{bare_home / '.config' / 'opencode'}:/home/dev/.config/opencode",
        f"{state}:/home/dev/.local/share/opencode",
    ]
    assert (state / "auth.json").read_text() == '{"llm": "key"}'
    assert not (state / "opencode.db").exists()  # seeded with auth ONLY
    assert (repo / ".task_worker" / ".gitignore").read_text() == "*\n"  # self-ignored


def test_run_once_preflight_fails_without_gaol_binary(tmp_path, monkeypatch):
    monkeypatch.setattr(sb.shutil, "which", lambda name: None)
    ready, status = GaolRunOnceSandbox(tmp_path).preflight()
    assert not ready
    assert "not on PATH" in status


def test_run_once_preflight_probes_run_once(tmp_path, monkeypatch):
    monkeypatch.setattr(sb.shutil, "which", lambda name: "/usr/bin/gaol")
    seen = _capture_run(monkeypatch)
    ready, status = GaolRunOnceSandbox(tmp_path).preflight()
    assert ready
    assert seen["args"] == ["gaol", "run-once", "--help"]


def test_run_once_run_tests_delegates_to_tester(tmp_path, monkeypatch):
    seen = {}

    def fake_run_tests(repo, sandbox=None):
        seen.update(repo=repo, sandbox=sandbox)
        return True, "green"

    monkeypatch.setattr(tester, "run_tests", fake_run_tests)
    box = GaolRunOnceSandbox(tmp_path)
    assert box.run_tests() == (True, "green")
    assert seen["sandbox"] is box


# --- factory ---------------------------------------------------------------------


def test_make_sandbox_default_is_gaol_dx(tmp_path):
    assert isinstance(make_sandbox(tmp_path), GaolDxSandbox)


def test_make_sandbox_kind_overrides_env_default(tmp_path):
    assert isinstance(make_sandbox(tmp_path, kind="gaol-run-once"), GaolRunOnceSandbox)
    assert isinstance(make_sandbox(tmp_path, kind="gaol-dx"), GaolDxSandbox)


def test_make_sandbox_env_can_select_run_once(tmp_path, monkeypatch):
    from forge.task_worker.config import settings

    monkeypatch.setattr(settings, "sandbox", "gaol-run-once")
    assert isinstance(make_sandbox(tmp_path), GaolRunOnceSandbox)


def test_make_sandbox_unknown_kind_raises(tmp_path, monkeypatch):
    from forge.task_worker.config import settings

    monkeypatch.setattr(settings, "sandbox", "warp-drive")
    with pytest.raises(ValueError, match="warp-drive"):
        make_sandbox(tmp_path)
    with pytest.raises(ValueError, match="warp-drive"):
        make_sandbox(tmp_path, kind="warp-drive")


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
    assert passed and out.startswith("no test runner configured")
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


def test_executor_quoted_spec_early_in_transcript_is_not_a_refusal(tmp_path):
    # The transcript includes the model READING the spec file, whose rules text can wrap
    # so that "BLOCKED:" lands at a line start — early occurrences must not count
    # (found by dogfood: a completed run was reverted as a false refusal).
    quoted_spec = (
        "reading spec...\nIf you cannot proceed, print a line starting with\nBLOCKED: and stop.\n"
    )
    work = "\n".join(f"[ok] step {i} done" for i in range(30))
    box = FakeSandbox(tmp_path, stdout=quoted_spec + work)
    ok, _, blocked = ex.execute_task_with_opencode(
        _task(), "SPEC", tmp_path, "auto", 60, sandbox=box
    )
    assert ok and not blocked


def test_executor_refusal_at_end_of_long_transcript_fires(tmp_path):
    work = "\n".join(f"[ok] step {i}" for i in range(30))
    box = FakeSandbox(tmp_path, stdout=work + "\nBLOCKED: dependency missing\n")
    ok, _, blocked = ex.execute_task_with_opencode(
        _task(), "SPEC", tmp_path, "auto", 60, sandbox=box
    )
    assert not ok and blocked


# --- rules layer: the repo's lessons ride in every worker prompt ----------------------


def test_write_spec_injects_repo_lessons_when_present(tmp_path):
    from forge.shared.lessons import append_lesson

    append_lesson(tmp_path, "always pin the toolchain version")
    spec_path = ex._write_spec(tmp_path, _task(), "DO THE TASK")
    content = spec_path.read_text()
    assert "always pin the toolchain version" in content
    assert "READ FIRST" in content
    # lessons come before the task body so the worker reads them first
    assert content.index("always pin the toolchain version") < content.index("DO THE TASK")


def test_write_spec_without_lessons_has_no_preamble(tmp_path):
    spec_path = ex._write_spec(tmp_path, _task(), "DO THE TASK")
    content = spec_path.read_text()
    assert "READ FIRST" not in content
    assert content.startswith(ex._SPEC_HEADER)
