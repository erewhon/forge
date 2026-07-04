"""run_one orchestration tests — every IO boundary (Nous, dx, VCS, tester, OpenCode) mocked.

The decision flow is under test: the fresh-read gate refusal, each failure path's distinct
reason, revert-before-status ordering, and the happy-path commit.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agents.task_worker import main as tw
from agents.task_worker.models import TaskInfo
from agents.task_worker.nous_client import _gate_reason


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


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Happy path across every boundary; returns an ordered event log for sequence asserts."""
    events: list = []
    (tmp_path / "Meta").mkdir()
    monkeypatch.setattr(tw.settings, "projects_dir", tmp_path)
    monkeypatch.setattr(tw.settings, "dry_run", False)
    monkeypatch.setattr(tw.settings, "default_max_files", 5)

    monkeypatch.setattr(tw, "check_worker_gate", lambda name: "")
    monkeypatch.setattr(tw, "get_task_spec", lambda name: "SPEC BODY")
    monkeypatch.setattr(tw, "detect_vcs", lambda p: "jj")
    fake_sandbox = SimpleNamespace(preflight=lambda: (True, "dx status: running"))
    monkeypatch.setattr(tw, "make_sandbox", lambda p: fake_sandbox)

    changed_calls = {"n": 0}

    def fake_changed(p):
        changed_calls["n"] += 1
        return [] if changed_calls["n"] == 1 else ["agents/x.py", "agents/tests/test_x.py"]

    monkeypatch.setattr(tw, "get_changed_files", fake_changed)
    monkeypatch.setattr(
        tw,
        "execute_task_with_opencode",
        lambda task, spec, pdir, model, timeout, sandbox=None: (
            events.append("execute") or (True, "ok", False)
        ),
    )
    monkeypatch.setattr(tw, "run_tests", lambda p, sandbox=None: (True, "all green"))
    monkeypatch.setattr(tw, "commit", lambda p, msg: events.append("commit") or "abc123")
    monkeypatch.setattr(tw, "revert_changes", lambda p: events.append("revert"))
    monkeypatch.setattr(
        tw,
        "update_task_status",
        lambda name, status, notes="": events.append(("status", status)),
    )
    return events


# --- gate ---------------------------------------------------------------------


def test_gate_refusal_skips_without_touching_anything(wired, monkeypatch):
    monkeypatch.setattr(tw, "check_worker_gate", lambda name: "status is 'Done', not Ready")
    out = tw.run_one(_task())
    assert out.status == "skipped"
    assert out.reason == "worker gate: status is 'Done', not Ready"
    assert wired == []  # no execute, no revert, no status write


def test_gate_check_error_fails_closed(wired, monkeypatch):
    def boom(name):
        raise RuntimeError("daemon down")

    monkeypatch.setattr(tw, "check_worker_gate", boom)
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
    assert "project dir not found" in out.reason


def test_dirty_working_copy_skips_before_in_progress(wired, monkeypatch):
    monkeypatch.setattr(tw, "get_changed_files", lambda p: ["already-dirty.py"])
    out = tw.run_one(_task())
    assert out.status == "skipped"
    assert "not clean" in out.reason
    assert wired == []  # In Progress never written


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


# --- failure paths ------------------------------------------------------------


def test_max_files_bail_reverts_before_reopening(wired, monkeypatch):
    out = tw.run_one(_task(max_files=1))  # post-exec diff has 2 files
    assert out.status == "failed"
    assert "max_files exceeded (2 > 1)" in out.reason
    assert out.changed_files == ["agents/x.py", "agents/tests/test_x.py"]
    assert "commit" not in wired
    # revert precedes the Ready write
    assert wired.index("revert") < wired.index(("status", "Ready"))


def test_tests_fail_reverts_and_reopens(wired, monkeypatch):
    monkeypatch.setattr(tw, "run_tests", lambda p, sandbox=None: (False, "assertion boom"))
    out = tw.run_one(_task())
    assert out.status == "failed"
    assert "tests failed" in out.reason
    assert "commit" not in wired
    assert wired.index("revert") < wired.index(("status", "Ready"))


def test_opencode_failure_without_diff_reverts_and_reopens(wired, monkeypatch):
    monkeypatch.setattr(
        tw, "execute_task_with_opencode", lambda *a, **k: (False, "model exploded", False)
    )
    monkeypatch.setattr(tw, "get_changed_files", lambda p: [])  # nothing left behind
    out = tw.run_one(_task())
    assert out.status == "failed"
    assert "opencode failed" in out.reason
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
    assert out.changed_files == ["agents/x.py", "agents/tests/test_x.py"]
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
    monkeypatch.setattr(tw, "find_next_task", lambda allowed: picked)
    seen = []
    monkeypatch.setattr(tw, "run_one", lambda task, **kw: seen.append(task))
    tw.run(project_filter="Meta")
    assert seen == [picked]


def test_run_no_ready_tasks_is_quiet_noop(wired, monkeypatch, capsys):
    monkeypatch.setattr(tw, "find_next_task", lambda allowed: None)
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
    notes_seen = []
    monkeypatch.setattr(
        tw,
        "update_task_status",
        lambda name, status, notes="": notes_seen.append(notes),
    )
    out = tw.run_one(_task())
    assert out.status == "failed"
    final_note = notes_seen[-1]
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
    monkeypatch.setattr(tw, "run_tests", lambda p, sandbox=None: tests_ran.append(1) or (True, ""))
    out = tw.run_one(_task())
    assert out.status == "failed"
    assert out.reason.startswith("lint failed:")
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
    assert "lint failed" in out.reason
