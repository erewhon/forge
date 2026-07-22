"""The drain loop — readiness filtering/ordering, the failing-head guard, dependency re-query, the
count cap, project scoping, and the unresolvable-title skip. Driven by a stateful fake store + fake
``run_one`` so no OpenCode/sandbox/Nous machinery is touched."""

from __future__ import annotations

from types import SimpleNamespace

from forge.queue.models import QueueRow
from forge.switcheroo.drain import drain, worker_ready_rows


def _row(project, task, *, mode="Auto-OK", priority=5, status="Ready", blocked=False):
    return QueueRow(
        project=project,
        task=task,
        status=status,
        execution_mode=mode,
        priority=priority,
        blocked=blocked,
    )


def _outcome(task, project, status, **kw):
    return SimpleNamespace(
        task=task,
        project=project,
        status=status,
        reason=kw.get("reason", ""),
        commit_id=kw.get("commit_id"),
        changed_files=kw.get("changed_files", []),
        duration_s=kw.get("duration_s", 0.0),
    )


class FakeStore:
    def __init__(self, rows):
        self.rows = rows

    def queue(self):
        # Mirrors the real store: Done tasks fall out of the queue.
        return [r for r in self.rows if r.status != "Done"]

    def find_task(self, name):
        for r in self.rows:
            if r.task == name:
                return SimpleNamespace(task=r.task, project=r.project)
        return None


def _make_run_one(store, *, calls, fail=(), unblock=None):
    """A fake worker: records each call, leaves failing leaves Ready (as the real revert does),
    marks successes Done, and optionally unblocks a dependent when its prerequisite lands."""

    def run_one_fn(task, *, repo=None):
        key = (task.project, task.task)
        calls.append(key)
        if key in fail:
            return _outcome(task.task, task.project, "failed", reason="boom")
        for r in store.rows:
            if (r.project, r.task) == key:
                r.status = "Done"
        if unblock and key in unblock:
            up, ut = unblock[key]
            for r in store.rows:
                if (r.project, r.task) == (up, ut):
                    r.blocked = False
        return _outcome(task.task, task.project, "done", commit_id=f"c-{task.task}")

    return run_one_fn


# --- readiness -------------------------------------------------------------


def test_worker_ready_filters_non_dispatchable_and_orders():
    store = FakeStore(
        [
            _row("P", "ready-ok"),
            _row("P", "manual", mode="Manual"),
            _row("P", "blocked", blocked=True),
            _row("P", "in-prog", status="In Progress"),
            _row("P", "preferred", mode="Auto-Preferred", priority=9),
            _row("P", "hi-prio", priority=1),
        ]
    )
    ready = worker_ready_rows(store, [])
    names = [r.task for r in ready]
    # Only dispatchable rows, Auto-Preferred first, then by ascending priority.
    assert names == ["preferred", "hi-prio", "ready-ok"]


def test_allowed_projects_narrows_scope():
    store = FakeStore([_row("P", "p1"), _row("Q", "q1")])
    assert [r.task for r in worker_ready_rows(store, ["P"])] == ["p1"]


# --- drain -----------------------------------------------------------------


def test_drains_each_dispatchable_once_in_order():
    store = FakeStore([_row("P", "a", priority=1), _row("P", "b", priority=2)])
    calls: list = []
    leaves = drain(store=store, allowed=[], run_one_fn=_make_run_one(store, calls=calls))
    assert calls == [("P", "a"), ("P", "b")]
    assert [(o.project, o.task, o.status) for o in leaves] == [
        ("P", "a", "done"),
        ("P", "b", "done"),
    ]
    assert leaves[0].commit_id == "c-a"


def test_failing_head_is_not_reselected():
    # 'bad' fails and reverts to Ready (stays dispatchable); it must be tried once, not forever,
    # and 'good' must still get drained.
    store = FakeStore([_row("P", "bad", priority=1), _row("P", "good", priority=2)])
    calls: list = []
    leaves = drain(
        store=store, allowed=[], run_one_fn=_make_run_one(store, calls=calls, fail={("P", "bad")})
    )
    assert calls == [("P", "bad"), ("P", "good")]
    assert {(o.task, o.status) for o in leaves} == {("bad", "failed"), ("good", "done")}


def test_dependency_unblocked_mid_window_is_picked_up():
    # 'dep' is blocked until 'first' lands; re-querying each pass must surface it.
    store = FakeStore([_row("P", "first", priority=1), _row("P", "dep", priority=2, blocked=True)])
    calls: list = []
    run_one = _make_run_one(store, calls=calls, unblock={("P", "first"): ("P", "dep")})
    leaves = drain(store=store, allowed=[], run_one_fn=run_one)
    assert calls == [("P", "first"), ("P", "dep")]
    assert all(o.status == "done" for o in leaves)


def test_max_leaves_caps_the_window():
    store = FakeStore([_row("P", "a", priority=1), _row("P", "b", priority=2), _row("P", "c")])
    calls: list = []
    leaves = drain(
        store=store, allowed=[], max_leaves=2, run_one_fn=_make_run_one(store, calls=calls)
    )
    assert len(leaves) == 2 and len(calls) == 2


def test_on_outcome_fires_per_leaf():
    store = FakeStore([_row("P", "a"), _row("P", "b")])
    seen: list = []
    drain(
        store=store,
        allowed=[],
        run_one_fn=_make_run_one(store, calls=[]),
        on_outcome=seen.append,
    )
    assert [o.task for o in seen] == ["a", "b"]


def test_unresolvable_title_is_skipped_not_run():
    # Same title in two projects: find_task resolves to the first (P), so the Q row can't be matched
    # to its own checkout and is recorded skipped rather than run against the wrong repo.
    store = FakeStore([_row("P", "dup", priority=1), _row("Q", "dup", priority=2)])
    calls: list = []
    leaves = drain(store=store, allowed=[], run_one_fn=_make_run_one(store, calls=calls))
    assert calls == [("P", "dup")]  # only the P row actually ran
    statuses = {(o.project, o.status) for o in leaves}
    assert statuses == {("P", "done"), ("Q", "skipped")}
