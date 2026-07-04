"""Dispatcher tests — run_one, find, and preflight mocked; lock behavior on a real tmp repo."""

from __future__ import annotations

import os

import pytest

from agents.coding_pipeline import dispatch as dp
from agents.coding_pipeline.dispatch import DispatchError, repo_lock, run_wave
from agents.coding_pipeline.models import WavePlan
from agents.task_worker.models import RunOutcome, TaskInfo


def _plan(*titles: str) -> WavePlan:
    return WavePlan(feature="Coding Pipeline", project="Meta", dispatch=list(titles))


def _task(title: str) -> TaskInfo:
    return TaskInfo(
        id=f"row-{title}",
        task=title,
        project="Meta",
        status="Ready",
        priority=3,
        execution_mode="Auto-OK",
    )


def _done(title: str) -> RunOutcome:
    return RunOutcome(
        task=title,
        project="Meta",
        status="done",
        commit_id="abc123",
        changed_files=["x.py"],
        duration_s=1.5,
    )


def _failed(title: str) -> RunOutcome:
    return RunOutcome(task=title, project="Meta", status="failed", reason="tests red")


@pytest.fixture
def wired(monkeypatch, tmp_path):
    monkeypatch.setattr(dp, "_preflight", lambda repo: "")
    return tmp_path


# --- wave execution -----------------------------------------------------------


def test_dispatches_serially_in_plan_order(wired):
    calls: list[str] = []

    def fake_run(task):
        calls.append(task.task)
        return _done(task.task)

    outcomes = run_wave(_plan("a", "b"), wired, run_leaf=fake_run, find=_task, log=lambda m: None)
    assert calls == ["a", "b"]
    assert [(o.leaf, o.status, o.commit_id) for o in outcomes] == [
        ("a", "done", "abc123"),
        ("b", "done", "abc123"),
    ]


def test_leaf_failure_does_not_abort_the_wave(wired):
    def fake_run(task):
        return _failed(task.task) if task.task == "a" else _done(task.task)

    outcomes = run_wave(_plan("a", "b"), wired, run_leaf=fake_run, find=_task, log=lambda m: None)
    assert [o.status for o in outcomes] == ["failed", "done"]


def test_missing_forge_task_is_skipped_and_wave_continues(wired):
    outcomes = run_wave(
        _plan("ghost", "real"),
        wired,
        run_leaf=lambda t: _done(t.task),
        find=lambda title: None if title == "ghost" else _task(title),
        log=lambda m: None,
    )
    assert outcomes[0].status == "skipped"
    assert "not found" in outcomes[0].reason
    assert outcomes[1].status == "done"


def test_outcomes_journaled_as_they_land(wired, tmp_path):
    journal_dir = tmp_path / "runs" / "epic"
    journal_dir.mkdir(parents=True)
    run_wave(
        _plan("a", "b"),
        wired,
        journal_dir=journal_dir,
        run_leaf=lambda t: _done(t.task),
        find=_task,
        log=lambda m: None,
    )
    lines = (journal_dir / "journal.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2  # one record per leaf, appended incrementally


def test_preflight_failure_aborts_with_nothing_dispatched(monkeypatch, tmp_path):
    monkeypatch.setattr(dp, "_preflight", lambda repo: "sandbox not ready: not created")
    calls: list[str] = []
    with pytest.raises(DispatchError, match="preflight"):
        run_wave(
            _plan("a"),
            tmp_path,
            run_leaf=lambda t: calls.append(t.task) or _done(t.task),
            find=_task,
            log=lambda m: None,
        )
    assert calls == []


# --- the repo lock ---------------------------------------------------------------


def test_lock_held_by_live_process_refuses_the_wave(wired):
    lock = wired / ".task_worker" / "dispatch.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text(str(os.getpid()))  # this very process: definitely alive
    with pytest.raises(DispatchError, match="another dispatch holds"):
        run_wave(_plan("a"), wired, run_leaf=_done, find=_task, log=lambda m: None)


def test_stale_lock_from_dead_process_is_stolen(wired, monkeypatch):
    lock = wired / ".task_worker" / "dispatch.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("12345")
    monkeypatch.setattr(dp, "_pid_alive", lambda pid: False)
    outcomes = run_wave(
        _plan("a"), wired, run_leaf=lambda t: _done(t.task), find=_task, log=lambda m: None
    )
    assert outcomes[0].status == "done"
    assert not lock.exists()  # released after the wave


def test_lock_released_even_when_a_leaf_raises(wired):
    def exploding_run(task):
        raise RuntimeError("worker crashed hard")

    with pytest.raises(RuntimeError, match="crashed hard"):
        run_wave(_plan("a"), wired, run_leaf=exploding_run, find=_task, log=lambda m: None)
    assert not (wired / ".task_worker" / "dispatch.lock").exists()


def test_unparseable_lock_is_treated_as_stale(wired):
    lock = wired / ".task_worker" / "dispatch.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("not-a-pid")
    outcomes = run_wave(
        _plan("a"), wired, run_leaf=lambda t: _done(t.task), find=_task, log=lambda m: None
    )
    assert outcomes[0].status == "done"


def test_repo_lock_context_manager_writes_own_pid(tmp_path):
    with repo_lock(tmp_path) as lock_path:
        assert int(lock_path.read_text()) == os.getpid()
    assert not lock_path.exists()
