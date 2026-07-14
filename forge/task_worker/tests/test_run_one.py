"""run_one orchestration tests — every IO boundary (Nous, dx, VCS, tester, OpenCode) mocked.

The decision flow is under test: the fresh-read gate refusal, each failure path's distinct
reason, revert-before-status ordering, and the happy-path commit.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from forge.task_worker import main as tw
from forge.task_worker.models import TaskInfo
from forge.task_worker.nous_client import _gate_reason


def _task(**overrides) -> TaskInfo:
    base = dict(
        id="row-1",
        task="Pipeline: wave planner",
        project="Meta",
        status="Ready",
        priority=2,
        execution_mode="Auto-OK",
        max_files=5,
        requires_tests=True,
    )
    base.update(overrides)
    return TaskInfo.model_validate(base)


class _FakeStore:
    """The worker's task backend under test. Tweak ``gate`` / ``gate_exc`` / ``spec`` /
    ``next_task`` per test; ``update_status`` records ('status', status) into the shared
    event log (for sequence asserts) and its notes into ``notes``."""

    def __init__(self, events: list) -> None:
        self.events = events
        self.gate = ""  # worker_gate return; non-"" refuses the leaf
        self.gate_exc: Exception | None = None  # set to raise from worker_gate
        self.spec = "SPEC BODY"
        self.next_task = None  # next_ready return
        self.notes: list[str] = []

    def worker_gate(self, name: str) -> str:
        if self.gate_exc is not None:
            raise self.gate_exc
        return self.gate

    def get_spec(self, name: str) -> str:
        return self.spec

    def update_status(self, task, status, notes="", execution_mode=None):
        self.events.append(("status", status))
        self.notes.append(notes)

    def next_ready(self, projects):
        return self.next_task


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Happy path across every boundary; returns an ordered event log for sequence asserts."""
    events: list = []
    store = _FakeStore(events)
    (tmp_path / "Meta").mkdir()
    monkeypatch.setattr(tw.settings, "projects_dir", tmp_path)
    monkeypatch.setattr(tw.settings, "dry_run", False)
    monkeypatch.setattr(tw.settings, "default_max_files", 5)

    monkeypatch.setattr(tw, "get_task_store", lambda: store)
    monkeypatch.setattr(tw, "detect_vcs", lambda p: "jj")
    fake_sandbox = SimpleNamespace(preflight=lambda: (True, "dx status: running"))
    monkeypatch.setattr(tw, "make_sandbox", lambda p: fake_sandbox)

    changed_calls = {"n": 0}

    def fake_changed(p):
        changed_calls["n"] += 1
        return [] if changed_calls["n"] == 1 else ["forge/x.py", "forge/tests/test_x.py"]

    monkeypatch.setattr(tw, "get_changed_files", fake_changed)
    monkeypatch.setattr(
        tw,
        "execute_task_with_opencode",
        lambda task, spec, pdir, model, timeout, sandbox=None: (
            events.append("execute") or (True, "ok", False)
        ),
    )
    monkeypatch.setattr(tw, "run_tests", lambda p, sandbox=None, **kw: (True, "all green"))
    monkeypatch.setattr(tw, "commit", lambda p, msg: events.append("commit") or "abc123")
    monkeypatch.setattr(tw, "revert_changes", lambda p: events.append("revert"))
    return events


# --- gate ---------------------------------------------------------------------


def test_gate_refusal_skips_without_touching_anything(wired, monkeypatch):
    tw.get_task_store().gate = "status is 'Done', not Ready"
    out = tw.run_one(_task())
    assert out.status == "skipped"
    assert out.reason == "worker gate: status is 'Done', not Ready"
    assert wired == []  # no execute, no revert, no status write


def test_gate_check_error_fails_closed(wired, monkeypatch):
    tw.get_task_store().gate_exc = RuntimeError("daemon down")
    out = tw.run_one(_task())
    assert out.status == "skipped"
    assert "gate check failed: daemon down" in out.reason
    assert wired == []


def test_gate_reason_pure_decision():
    assert _gate_reason("Ready", "Auto-OK", []) == ""
    assert _gate_reason("Ready", "Auto-Preferred", []) == ""
    assert "not Ready" in _gate_reason("Spec Needed", "Auto-OK", [])
    assert "'Manual'" in _gate_reason("Ready", "", [])  # null-as-manual
    assert "'Manual'" in _gate_reason("Ready", "Manual", [])
    assert "unmet dependencies: dep-a" in _gate_reason("Ready", "Auto-OK", ["dep-a"])


# --- preflights ---------------------------------------------------------------


def test_project_dir_falls_back_to_lowercase(wired, monkeypatch, tmp_path):
    # Forge project "Meta" ↔ checkout dir "meta": only the lowercase dir exists.
    lower_root = tmp_path / "lower"
    (lower_root / "meta").mkdir(parents=True)
    monkeypatch.setattr(tw.settings, "projects_dir", lower_root)
    out = tw.run_one(_task())
    assert out.status == "done"  # resolved to the lowercase dir and ran through


def test_missing_project_dir_skips(wired, monkeypatch, tmp_path):
    monkeypatch.setattr(tw.settings, "projects_dir", tmp_path / "nowhere")
    out = tw.run_one(_task())
    assert out.status == "skipped"
    assert "could not locate a checkout" in out.reason


# --- project-dir resolution (cwd-first) ---------------------------------------


def test_project_key_normalizes_case_and_separators():
    assert tw._project_key("Meta") == tw._project_key("meta")
    assert tw._project_key("soft-serve-with-sprinkles") == tw._project_key(
        "soft_serve_with_sprinkles"
    )


def test_find_repo_root_walks_up(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    deep = repo / "src" / "pkg"
    deep.mkdir(parents=True)
    assert tw._find_repo_root(deep) == repo

    bare = tmp_path / "no-vcs" / "here"
    bare.mkdir(parents=True)
    assert tw._find_repo_root(bare) is None


def test_resolve_prefers_cwd_when_name_matches(monkeypatch, tmp_path):
    repo = tmp_path / "Meta"
    repo.mkdir()
    monkeypatch.setattr(tw, "_find_repo_root", lambda start: repo)
    monkeypatch.setattr(tw.settings, "projects_dir", tmp_path / "nowhere")
    assert tw._resolve_project_dir(_task()) == repo


def test_resolve_matches_cwd_case_insensitively(monkeypatch, tmp_path):
    repo = tmp_path / "meta"  # lowercase dir vs Title-case project
    repo.mkdir()
    monkeypatch.setattr(tw, "_find_repo_root", lambda start: repo)
    monkeypatch.setattr(tw.settings, "projects_dir", tmp_path / "nowhere")
    assert tw._resolve_project_dir(_task()) == repo


def test_resolve_falls_back_to_base_when_cwd_is_other_repo(monkeypatch, tmp_path):
    other = tmp_path / "some-other-repo"
    other.mkdir()
    base = tmp_path / "base"
    (base / "Meta").mkdir(parents=True)
    monkeypatch.setattr(tw, "_find_repo_root", lambda start: other)
    monkeypatch.setattr(tw.settings, "projects_dir", base)
    # cwd is a repo but not this project's — must NOT commit there; use the base checkout.
    assert tw._resolve_project_dir(_task()) == base / "Meta"


def test_resolve_returns_none_when_cwd_mismatch_and_no_base(monkeypatch, tmp_path):
    other = tmp_path / "some-other-repo"
    other.mkdir()
    monkeypatch.setattr(tw, "_find_repo_root", lambda start: other)
    monkeypatch.setattr(tw.settings, "projects_dir", tmp_path / "nowhere")
    assert tw._resolve_project_dir(_task()) is None


def test_dirty_working_copy_skips_before_in_progress(wired, monkeypatch):
    monkeypatch.setattr(tw, "get_changed_files", lambda p: ["already-dirty.py"])
    out = tw.run_one(_task())
    assert out.status == "skipped"
    assert "not clean" in out.reason
    assert wired == []  # In Progress never written


def test_dx_not_ready_carries_the_remedy_in_the_reason(wired, monkeypatch):
    monkeypatch.setattr(
        tw,
        "make_sandbox",
        lambda *a, **k: SimpleNamespace(preflight=lambda: (False, "not created")),
    )
    out = tw.run_one(_task())
    assert out.status == "skipped"
    # the fix travels in the reason the loop/escalation consumes, not just the console print
    assert "gaol dx shell" in out.reason
    assert "not created" in out.reason


def test_gate_failure_reason_names_expectation_then_evidence():
    msg = tw._gate_failure("tests", "expected the suite to pass", "3 failed")
    assert msg.startswith("tests gate failed — expected the suite to pass.")
    assert "Evidence:\n3 failed" in msg


def test_blocked_marker_in_spec_skips(wired):
    out = tw.run_one(_task(), spec="> **Blocked:** deps not Done")
    assert out.status == "skipped"
    assert out.reason == "spec contains BLOCKED marker"


def test_blocked_line_at_start_skips(wired):
    out = tw.run_one(_task(), spec="notes\nBLOCKED: cannot proceed\nmore")
    assert out.status == "skipped"


def test_mentioning_blocked_protocol_mid_line_does_not_skip(wired):
    # A respec that documents the protocol must not trip the marker (found by dogfood).
    spec = "Do the task. If stuck, print a line starting with `BLOCKED:` and stop."
    out = tw.run_one(_task(), spec=spec)
    assert out.status == "done"


# --- repo override (concurrent dispatch runs leaves inside jj workspaces) -------


def test_repo_override_pins_every_collaborator_to_the_same_dir(wired, monkeypatch, tmp_path):
    """SAFETY: the clean-WC guard, sandbox, executor, lint, tests, revert, and commit must
    all act on the override dir — a mixed state (gate on the workspace, commit on the
    settings path) would corrupt the host checkout."""
    workspace = tmp_path / "cw-leaf-abc123"
    workspace.mkdir()
    # The conventional resolution must never be consulted: point it somewhere nonexistent.
    monkeypatch.setattr(tw.settings, "projects_dir", tmp_path / "nowhere")

    boundaries = ("vcs", "sandbox", "changed", "exec", "lint", "tests", "commit")
    seen: dict[str, list] = {k: [] for k in boundaries}
    monkeypatch.setattr(tw, "detect_vcs", lambda p: seen["vcs"].append(p) or "jj")
    fake_sandbox = SimpleNamespace(preflight=lambda: (True, "ok"))
    monkeypatch.setattr(tw, "make_sandbox", lambda p: seen["sandbox"].append(p) or fake_sandbox)

    changed_calls = {"n": 0}

    def fake_changed(p):
        seen["changed"].append(p)
        changed_calls["n"] += 1
        return [] if changed_calls["n"] == 1 else ["forge/x.py"]

    monkeypatch.setattr(tw, "get_changed_files", fake_changed)
    monkeypatch.setattr(
        tw,
        "execute_task_with_opencode",
        lambda task, spec, pdir, model, timeout, sandbox=None: (
            seen["exec"].append(pdir) or (True, "ok", False)
        ),
    )
    monkeypatch.setattr(
        tw, "run_lint", lambda p, files, sandbox=None: seen["lint"].append(p) or (True, "", False)
    )
    monkeypatch.setattr(
        tw, "run_tests", lambda p, sandbox=None, **kw: seen["tests"].append(p) or (True, "green")
    )
    monkeypatch.setattr(tw, "commit", lambda p, msg: seen["commit"].append(p) or "abc123")

    out = tw.run_one(_task(), repo=workspace)
    assert out.status == "done"
    for boundary, paths in seen.items():
        assert paths, f"{boundary} never called"
        assert all(p == workspace for p in paths), f"{boundary} saw {paths}, not the override"


def test_missing_repo_override_skips_without_falling_back(wired, monkeypatch, tmp_path):
    # The override is authoritative: a missing workspace must NOT fall back to the
    # conventional checkout (which exists here and would silently absorb the leaf).
    out = tw.run_one(_task(), repo=tmp_path / "gone")
    assert out.status == "skipped"
    assert "repo override not found" in out.reason
    assert wired == []  # nothing executed, no status writes


def test_no_override_keeps_conventional_resolution(wired):
    out = tw.run_one(_task())  # the wired fixture's projects_dir/Meta path
    assert out.status == "done"  # byte-for-byte the pre-override behavior


def test_sandbox_kind_passes_through_to_factory(wired, monkeypatch):
    kinds = []
    fake_sandbox = SimpleNamespace(preflight=lambda: (True, "ok"))

    def fake_factory(p, kind=None):
        kinds.append(kind)
        return fake_sandbox

    monkeypatch.setattr(tw, "make_sandbox", fake_factory)
    out = tw.run_one(_task(), sandbox_kind="gaol-run-once")
    assert out.status == "done"
    assert kinds == ["gaol-run-once"]


def test_no_sandbox_kind_calls_factory_without_kind(wired, monkeypatch):
    """None must stay byte-for-byte today's single-argument call — the factory only grows
    its ``kind`` parameter in the run-once leaf, and a kind=None pass-through would crash
    against the current signature."""
    calls = []
    fake_sandbox = SimpleNamespace(preflight=lambda: (True, "ok"))

    def fake_factory(p):  # today's signature: kind would TypeError
        calls.append(p)
        return fake_sandbox

    monkeypatch.setattr(tw, "make_sandbox", fake_factory)
    out = tw.run_one(_task())
    assert out.status == "done"
    assert len(calls) == 1


# --- failure paths ------------------------------------------------------------


def test_max_files_bail_reverts_before_reopening(wired, monkeypatch):
    # requires_tests=False so the requires_tests floor (>=3) stays out of the way —
    # this test is about the bail mechanics, not the floor.
    out = tw.run_one(_task(max_files=1, requires_tests=False))  # post-exec diff has 2 files
    assert out.status == "failed"
    assert "max_files exceeded (2 > 1)" in out.reason
    assert out.changed_files == ["forge/x.py", "forge/tests/test_x.py"]
    assert "commit" not in wired
    # revert precedes the Ready write
    assert wired.index("revert") < wired.index(("status", "Ready"))


def test_tests_fail_reverts_and_reopens(wired, monkeypatch):
    monkeypatch.setattr(tw, "run_tests", lambda p, sandbox=None, **kw: (False, "assertion boom"))
    out = tw.run_one(_task())
    assert out.status == "failed"
    assert "tests gate failed" in out.reason  # names the failed expectation for the next attempt
    assert "assertion boom" in out.reason  # ...and carries the evidence
    assert "commit" not in wired
    assert wired.index("revert") < wired.index(("status", "Ready"))


def test_opencode_failure_without_diff_reverts_and_reopens(wired, monkeypatch):
    monkeypatch.setattr(
        tw, "execute_task_with_opencode", lambda *a, **k: (False, "model exploded", False)
    )
    monkeypatch.setattr(tw, "get_changed_files", lambda p: [])  # nothing left behind
    out = tw.run_one(_task())
    assert out.status == "failed"
    assert "opencode gate failed" in out.reason
    assert "model exploded" in out.reason
    assert wired.index("revert") < wired.index(("status", "Ready"))


def test_opencode_nonzero_exit_with_diff_proceeds_to_gates(wired, monkeypatch):
    # Session-end plugin crashes fail the process after the model finished (observed:
    # open-mem missing its API key). Exit code is advisory — the gates decide.
    monkeypatch.setattr(
        tw, "execute_task_with_opencode", lambda *a, **k: (False, "plugin died at exit", False)
    )
    out = tw.run_one(_task())
    assert out.status == "done"  # scope + tests + commit all ran and passed
    assert out.commit_id == "abc123"
    assert "revert" not in wired


def test_model_blocked_refusal_always_reverts(wired, monkeypatch):
    # An explicit BLOCKED refusal reverts even though a diff exists — never gate-check
    # work the model itself disowned.
    monkeypatch.setattr(
        tw, "execute_task_with_opencode", lambda *a, **k: (False, "BLOCKED: missing dep", True)
    )
    out = tw.run_one(_task())
    assert out.status == "failed"
    assert "BLOCKED" in out.reason
    assert "commit" not in wired
    assert wired.index("revert") < wired.index(("status", "Ready"))


# --- happy path + dry run -------------------------------------------------------


def test_happy_path_commits_and_marks_done(wired):
    out = tw.run_one(_task())
    assert out.status == "done"
    assert out.commit_id == "abc123"
    assert out.changed_files == ["forge/x.py", "forge/tests/test_x.py"]
    assert out.notes_written is True
    assert out.duration_s >= 0
    assert ("status", "In Progress") in wired
    assert wired.index("commit") < wired.index(("status", "Done"))
    assert "revert" not in wired


def test_dry_run_executes_then_reverts_without_nous_writes(wired, monkeypatch):
    monkeypatch.setattr(tw.settings, "dry_run", True)
    out = tw.run_one(_task())
    assert out.status == "skipped"
    assert "dry-run" in out.reason
    assert out.notes_written is False
    assert "execute" in wired
    assert "revert" in wired
    assert "commit" not in wired
    assert not any(isinstance(e, tuple) and e[0] == "status" for e in wired)


# --- CLI wrapper ----------------------------------------------------------------


def test_run_delegates_selection_to_run_one(wired, monkeypatch):
    picked = _task()
    tw.get_task_store().next_task = picked
    seen = []
    monkeypatch.setattr(tw, "run_one", lambda task, **kw: seen.append(task))
    tw.run(project_filter="Meta")
    assert seen == [picked]


def test_run_no_ready_tasks_is_quiet_noop(wired, monkeypatch, capsys):
    tw.get_task_store().next_task = None  # store's default: nothing ready
    tw.run()
    assert "No worker-ready tasks found" in capsys.readouterr().out


# --- degenerate-session retry + no-change diagnostics (dry-run findings) ------------


def test_degenerate_session_retries_in_process_before_failing(wired, monkeypatch):
    """A near-instant session with zero file changes is an empty generation, not a
    real attempt (observed live: 2.8s, zero tool calls, burned the leaf's last
    attempt) — it retries in-process before the failure is recorded."""
    monkeypatch.setattr(tw.settings, "degenerate_retries", 1)
    monkeypatch.setattr(tw.settings, "degenerate_session_seconds", 10.0)
    monkeypatch.setattr(tw, "get_changed_files", lambda p: [])  # nothing ever changes
    executes = {"n": 0}
    monkeypatch.setattr(
        tw,
        "execute_task_with_opencode",
        lambda *a, **k: executes.__setitem__("n", executes["n"] + 1) or (True, "empty", False),
    )
    out = tw.run_one(_task())
    assert executes["n"] == 2  # original + one in-process retry
    assert out.status == "failed"
    assert out.reason == "no file changes produced"


def test_slow_no_change_session_does_not_retry(wired, monkeypatch):
    """Only FAST empty sessions are degenerate — a session that ran long and produced
    nothing is a real (failed) attempt, not a router hiccup."""
    monkeypatch.setattr(tw.settings, "degenerate_retries", 1)
    monkeypatch.setattr(tw.settings, "degenerate_session_seconds", 0.0)  # everything is "slow"
    monkeypatch.setattr(tw, "get_changed_files", lambda p: [])
    executes = {"n": 0}
    monkeypatch.setattr(
        tw,
        "execute_task_with_opencode",
        lambda *a, **k: executes.__setitem__("n", executes["n"] + 1) or (True, "empty", False),
    )
    out = tw.run_one(_task())
    assert executes["n"] == 1
    assert out.status == "failed"


def test_blocked_session_never_retries(wired, monkeypatch):
    monkeypatch.setattr(tw.settings, "degenerate_retries", 1)
    executes = {"n": 0}
    monkeypatch.setattr(
        tw,
        "execute_task_with_opencode",
        lambda *a, **k: (
            executes.__setitem__("n", executes["n"] + 1) or (False, "BLOCKED: cannot", True)
        ),
    )
    out = tw.run_one(_task())
    assert executes["n"] == 1  # an explicit refusal is a verdict, not a hiccup
    assert "BLOCKED" in out.reason


def test_no_change_failure_notes_carry_session_tail(wired, monkeypatch):
    """Triage used to require container-side opencode logs — the task note now carries
    the session tail like the tests-failed path does."""
    monkeypatch.setattr(tw.settings, "degenerate_retries", 0)
    monkeypatch.setattr(tw, "get_changed_files", lambda p: [])
    monkeypatch.setattr(
        tw,
        "execute_task_with_opencode",
        lambda *a, **k: (True, "model said: I analyzed the code thoroughly", False),
    )
    out = tw.run_one(_task())
    assert out.status == "failed"
    final_note = tw.get_task_store().notes[-1]
    assert "Session tail" in final_note
    assert "analyzed the code thoroughly" in final_note


def test_tail_trims_to_line_boundary():
    text = "\n".join(f"line {i:03d} " + "x" * 40 for i in range(40))
    tail = tw._tail(text, 200)
    assert len(tail) <= 200
    assert tail.startswith("line ")  # opens on a whole line, never mid-line
    assert tail.endswith(text[-20:])

    one_long_line = "y" * 1000
    assert tw._tail(one_long_line, 200) == "y" * 200  # no newline: mid-line beats empty

    short = "short text"
    assert tw._tail(short, 200) == short


# --- lint gate (autofix-then-recheck, before tests) ----------------------------------


def test_lint_failure_reverts_before_tests(wired, monkeypatch):
    events = wired
    monkeypatch.setattr(
        tw, "run_lint", lambda p, files, sandbox=None: (False, "E501 survived autofix", True)
    )
    tests_ran = []
    monkeypatch.setattr(
        tw, "run_tests", lambda p, sandbox=None, **kw: tests_ran.append(1) or (True, "")
    )
    out = tw.run_one(_task())
    assert out.status == "failed"
    assert out.reason.startswith("lint gate failed —")
    assert tests_ran == []  # lint gates BEFORE tests: one test run per leaf, final state only
    assert "revert" in events  # reverted like any other gate failure
    assert ("status", "Ready") in events


def test_lint_autofix_then_clean_proceeds_to_commit(wired, monkeypatch):
    events = wired
    monkeypatch.setattr(
        tw, "run_lint", lambda p, files, sandbox=None: (True, "lint clean after autofix", True)
    )
    out = tw.run_one(_task())
    assert out.status == "done"
    assert "commit" in events  # autofixed state landed


def test_lint_skipped_when_tests_not_required(wired, monkeypatch):
    lint_calls = []
    monkeypatch.setattr(
        tw, "run_lint", lambda p, files, sandbox=None: lint_calls.append(1) or (True, "", False)
    )
    out = tw.run_one(_task(requires_tests=False))
    assert out.status == "done"
    assert lint_calls == []  # gated on requires_tests, per the task spec


def test_linter_crash_fails_closed(wired, monkeypatch):
    def boom(p, files, sandbox=None):
        raise RuntimeError("uvx exploded")

    monkeypatch.setattr(tw, "run_lint", boom)
    out = tw.run_one(_task())
    assert out.status == "failed"
    assert "lint gate failed" in out.reason
    assert "uvx exploded" in out.reason  # the crash detail survives into the actionable reason


def test_requires_tests_floors_max_files_at_three(wired):
    # A requires_tests leaf with max_files < 3 is structurally impossible (impl +
    # test >= 2 files); the worker floors the effective budget so a too-tight
    # decomposition can't revert correct work. The 2-file diff passes under the floor.
    out = tw.run_one(_task(max_files=1, requires_tests=True))
    assert "max_files exceeded" not in out.reason
