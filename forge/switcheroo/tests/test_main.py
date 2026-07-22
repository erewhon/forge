"""`switcheroo_now` orchestration — the baton gate, dry-run re-anchor, and a full drained window
writing its journal. Fakes stand in for the Nous store and the OpenCode worker."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from forge.queue.models import QueueRow
from forge.shared.baton import Baton, read_baton, write_baton
from forge.switcheroo.journal import (
    end_failover,
    failover_path,
    history_dir,
    read_failover,
    record_outcome,
    start_failover,
)
from forge.switcheroo.main import switch_back, switcheroo_now
from forge.switcheroo.models import LeafOutcome


def _row(project, task, **kw):
    return QueueRow(
        project=project,
        task=task,
        status=kw.get("status", "Ready"),
        execution_mode=kw.get("mode", "Auto-OK"),
        priority=kw.get("priority", 5),
        blocked=kw.get("blocked", False),
    )


class FakeStore:
    def __init__(self, rows):
        self.rows = rows

    def queue(self):
        return [r for r in self.rows if r.status != "Done"]

    def find_task(self, name):
        for r in self.rows:
            if r.task == name:
                return SimpleNamespace(task=r.task, project=r.project)
        return None


def _run_one(store):
    def fn(task, *, repo=None):
        for r in store.rows:
            if (r.project, r.task) == (task.project, task.task):
                r.status = "Done"
        return SimpleNamespace(
            task=task.task,
            project=task.project,
            status="done",
            reason="",
            commit_id=f"c-{task.task}",
            changed_files=["f.py"],
            duration_s=1.0,
        )

    return fn


# --- the baton gate --------------------------------------------------------


def test_aborts_when_no_baton_and_no_goal(tmp_path: Path):
    rc = switcheroo_now(home=tmp_path, store=FakeStore([]), run_one_fn=lambda *a, **k: None)
    assert rc == 1
    assert not failover_path(tmp_path).is_file()  # no window opened


def test_synthesizes_baton_from_goal(tmp_path: Path):
    rc = switcheroo_now(home=tmp_path, goal="finish the migration", store=FakeStore([]))
    assert rc == 0
    baton = read_baton(tmp_path)
    assert baton is not None and baton.goal == "finish the migration"


def test_goal_arg_updates_existing_baton(tmp_path: Path):
    write_baton(tmp_path, Baton(goal="old goal", decisions=["keep me"]))
    switcheroo_now(home=tmp_path, goal="new goal", store=FakeStore([]))
    baton = read_baton(tmp_path)
    assert baton.goal == "new goal"
    assert baton.decisions == ["keep me"]  # decision accretion preserved through the re-anchor


# --- dry run ---------------------------------------------------------------


def test_dry_run_reanchors_but_runs_nothing(tmp_path: Path):
    write_baton(tmp_path, Baton(goal="g"))
    store = FakeStore([_row("P", "a")])
    called = []
    rc = switcheroo_now(
        home=tmp_path,
        dry_run=True,
        store=store,
        run_one_fn=lambda *a, **k: called.append(1),
    )
    assert rc == 0
    assert called == []  # no leaf ran
    assert not failover_path(tmp_path).is_file()  # no window opened


# --- a full window ---------------------------------------------------------


def test_no_ready_leaves_opens_no_window(tmp_path: Path):
    write_baton(tmp_path, Baton(goal="g"))
    rc = switcheroo_now(home=tmp_path, store=FakeStore([]), run_one_fn=lambda *a, **k: None)
    assert rc == 0
    assert not failover_path(tmp_path).is_file()


def test_drains_ready_leaves_and_writes_closed_journal(tmp_path: Path):
    write_baton(tmp_path, Baton(goal="g"))
    store = FakeStore([_row("P", "a", priority=1), _row("Q", "b", priority=2)])
    rc = switcheroo_now(
        home=tmp_path, reason="all agents down", store=store, run_one_fn=_run_one(store)
    )
    assert rc == 0
    log = read_failover(tmp_path)
    assert log is not None
    assert log.ended_at is not None  # window closed
    assert log.reason == "all agents down"
    assert [(o.project, o.task, o.status) for o in log.outcomes] == [
        ("P", "a", "done"),
        ("Q", "b", "done"),
    ]
    assert log.done[0].commit_id == "c-a"


# --- switch-back -----------------------------------------------------------


def _seed_window(home: Path) -> None:
    baton = write_baton(home, Baton(goal="ship it", decisions=["keep me"]))
    start_failover(home, baton=baton, model_tier="auto", reason="outage")
    record_outcome(home, LeafOutcome(task="t1", project="P", status="done", commit_id="c1"))
    end_failover(home)


def test_switch_back_nothing_to_do(tmp_path: Path):
    assert switch_back(home=tmp_path) == 0
    assert not failover_path(tmp_path).is_file()


def test_switch_back_dry_run_mutates_nothing(tmp_path: Path):
    _seed_window(tmp_path)
    before = read_baton(tmp_path).model_dump()
    assert switch_back(home=tmp_path, dry_run=True) == 0
    # Window still active, baton untouched.
    assert failover_path(tmp_path).is_file()
    assert read_baton(tmp_path).model_dump() == before


def test_switch_back_reanchors_baton_and_archives_window(tmp_path: Path):
    _seed_window(tmp_path)
    assert switch_back(home=tmp_path) == 0

    # Window consumed → archived, no longer active.
    assert not failover_path(tmp_path).is_file()
    assert len(list(history_dir(tmp_path).glob("*.json"))) == 1

    baton = read_baton(tmp_path)
    assert baton.next_action.startswith("Reconcile the failover window")
    assert "keep me" in baton.decisions  # prior decisions preserved (accretion)
    assert any("Failover window" in d for d in baton.decisions)  # window noted


def test_switch_back_closes_an_interrupted_window(tmp_path: Path):
    baton = write_baton(tmp_path, Baton(goal="g"))
    start_failover(tmp_path, baton=baton, model_tier="auto")
    record_outcome(tmp_path, LeafOutcome(task="t1", project="P", status="done"))
    # Note: no end_failover() — the window is left open (interrupted).
    assert read_failover(tmp_path).ended_at is None

    assert switch_back(home=tmp_path) == 0
    # Consumed and archived; the archived copy has an ended_at stamped on switch-back.
    archived = list(history_dir(tmp_path).glob("*.json"))
    assert len(archived) == 1
    import json

    assert json.loads(archived[0].read_text())["ended_at"] is not None
