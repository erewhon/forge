"""Dispatcher tests — run_one, find, and preflight mocked; lock behavior on a real tmp repo."""

from __future__ import annotations

import os
import threading
import time
from types import SimpleNamespace

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
    """Serial-path harness: preflight mocked AND concurrency pinned to 1 — the shipped
    default is 3 (post-smoke), and these tests exercise the serial dispatcher."""
    monkeypatch.setattr(dp, "_preflight", lambda repo: "")
    monkeypatch.setattr(dp.settings, "dispatch_concurrency", 1)
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
    monkeypatch.setattr(dp.settings, "dispatch_concurrency", 1)
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


# --- epic-context injection (sibling contracts) --------------------------------


def _task_with_deps(title: str, deps: list[str]) -> TaskInfo:
    t = _task(title)
    return t.model_copy(update={"deps": deps})


def test_preamble_prepended_to_fetched_spec(wired):
    seen: dict = {}

    def fake_run(task, spec=None):
        seen["spec"] = spec
        return _done(task.task)

    run_wave(
        _plan("a"),
        wired,
        run_leaf=fake_run,
        find=_task,
        preamble_for=lambda task: "## Epic context\ncontracts here",
        fetch_spec=lambda title: "## Task: a\nthe spec body",
        log=lambda m: None,
    )
    assert seen["spec"].startswith("## Epic context")
    assert "the spec body" in seen["spec"]


def test_empty_preamble_dispatches_plain(wired):
    seen: dict = {}

    def fake_run(task, spec=None):
        seen["spec"] = spec
        return _done(task.task)

    run_wave(
        _plan("a"),
        wired,
        run_leaf=fake_run,
        find=_task,
        preamble_for=lambda task: "",
        fetch_spec=lambda title: (_ for _ in ()).throw(AssertionError("no fetch on empty")),
        log=lambda m: None,
    )
    assert seen["spec"] is None


def test_preamble_failure_never_blocks_dispatch(wired, tmp_path):
    """Context injection is best-effort by design: a crashing builder logs, journals
    the error, and the leaf still runs plain."""

    def boom(task):
        raise RuntimeError("journal unreadable")

    outcomes = run_wave(
        _plan("a"),
        wired,
        journal_dir=tmp_path,
        run_leaf=lambda t, spec=None: _done(t.task),
        find=_task,
        preamble_for=boom,
        log=lambda m: None,
    )
    assert outcomes[0].status == "done"
    journal = (tmp_path / "journal.jsonl").read_text()
    assert "leaf_context" in journal and "journal unreadable" in journal


def test_spec_fetch_failure_degrades_to_plain_dispatch(wired):
    seen: dict = {}

    def fake_run(task, spec=None):
        seen["spec"] = spec
        return _done(task.task)

    def failing_fetch(title):
        raise ConnectionError("daemon down")

    outcomes = run_wave(
        _plan("a"),
        wired,
        run_leaf=fake_run,
        find=_task,
        preamble_for=lambda task: "## Epic context\nstuff",
        fetch_spec=failing_fetch,
        log=lambda m: None,
    )
    assert outcomes[0].status == "done"
    assert seen["spec"] is None  # worker fetches its own spec


def test_injection_journaled_with_deps_and_size(wired, tmp_path):
    import json as _json

    run_wave(
        _plan("a"),
        wired,
        journal_dir=tmp_path,
        run_leaf=lambda t, spec=None: _done(t.task),
        find=lambda title: _task_with_deps(title, ["landed dep", "unlanded dep"]),
        preamble_for=lambda task: '## Epic context\n- "landed dep" (commit abc):\n  - x.py',
        fetch_spec=lambda title: "spec",
        log=lambda m: None,
    )
    records = [
        _json.loads(line)
        for line in (tmp_path / "journal.jsonl").read_text().splitlines()
        if '"leaf_context"' in line
    ]
    assert len(records) == 1
    assert records[0]["deps_landed"] == ["landed dep"]
    assert records[0]["chars"] > 0


# --- concurrent dispatch (workspace fan-out + reconcile barrier) --------------------


def _done_at(title: str) -> RunOutcome:
    return _done(title).model_copy(update={"commit_id": f"cid-{title}"})


@pytest.fixture
def cw(monkeypatch, tmp_path):
    """Fake every seam the concurrent path uses: workspace lifecycle, reconcile, task
    store. Records what happened so tests assert behavior, not wiring trivia."""
    import agents.coding_pipeline.reconcile as rcmod
    import agents.shared.task_store as tsmod
    import agents.shared.workspaces as wsmod

    state = SimpleNamespace(
        created=[],  # (dest, base_rev)
        forgotten=[],
        demotes=[],  # (task, status)
        reconcile_landed=None,  # [(leaf, commit_id)] as received, in order
        reconcile_result=None,  # override to script demotions
        preflight_kinds=[],
    )

    def fake_preflight(repo, kind=None):
        state.preflight_kinds.append(kind)
        return ""

    monkeypatch.setattr(dp, "_preflight", fake_preflight)
    monkeypatch.setattr(wsmod, "resolve_base_rev", lambda repo, rev="@": "base0")
    monkeypatch.setattr(
        wsmod,
        "workspace_destination",
        lambda repo, label, base_dir=None, prefix="ws": tmp_path / f"{prefix}-{label}",
    )

    def fake_create(repo, dest, *, base_rev):
        state.created.append((dest, base_rev))
        dest.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(wsmod, "create_workspace", fake_create)
    monkeypatch.setattr(wsmod, "forget_workspace", lambda repo, ws: state.forgotten.append(ws))

    def fake_reconcile(repo, base, landed, *, on_demote, log):
        state.reconcile_landed = [(o.leaf, c) for o, c in landed]
        if state.reconcile_result is not None:
            return state.reconcile_result(landed, on_demote)
        return [o for o, _ in landed]

    monkeypatch.setattr(rcmod, "reconcile_wave", fake_reconcile)

    class _Store:
        def update_status(self, task, status, notes="", **kw):
            state.demotes.append((task, status))

        def find_task(self, title):  # only reached when a test omits find=
            return _task(title)

        def get_spec(self, title):  # only reached when a test omits fetch_spec=
            return "spec"

    monkeypatch.setattr(tsmod, "get_task_store", lambda: _Store())
    return state


def test_concurrent_runs_each_leaf_in_its_own_workspace(cw, tmp_path):
    seen: dict[str, dict] = {}

    def fake_run(task, spec=None, repo=None, sandbox_kind=None):
        seen[task.task] = {"repo": repo, "sandbox_kind": sandbox_kind}
        return _done_at(task.task)

    outcomes = run_wave(
        _plan("a", "b", "c"),
        tmp_path,
        run_leaf=fake_run,
        find=_task,
        concurrency=3,
        log=lambda m: None,
    )
    assert [o.status for o in outcomes] == ["done", "done", "done"]
    # one workspace per leaf, all pinned to the SAME base rev
    assert len(cw.created) == 3
    assert {base for _, base in cw.created} == {"base0"}
    # each worker ran against ITS workspace, under the run-once sandbox
    workspace_paths = {dest for dest, _ in cw.created}
    assert {v["repo"] for v in seen.values()} == workspace_paths
    assert {v["sandbox_kind"] for v in seen.values()} == {"gaol-run-once"}
    # preflight probed the sandbox kind the wave actually runs under
    assert cw.preflight_kinds == ["gaol-run-once"]
    # every workspace forgotten at the end
    assert sorted(cw.forgotten) == sorted(workspace_paths)


def test_concurrent_cap_bounds_leaves_in_flight(cw, tmp_path):
    lock = threading.Lock()
    gauge = {"now": 0, "max": 0}

    def fake_run(task, spec=None, repo=None, sandbox_kind=None):
        with lock:
            gauge["now"] += 1
            gauge["max"] = max(gauge["max"], gauge["now"])
        time.sleep(0.05)
        with lock:
            gauge["now"] -= 1
        return _done_at(task.task)

    run_wave(
        _plan("a", "b", "c", "d"),
        tmp_path,
        run_leaf=fake_run,
        find=_task,
        concurrency=2,
        log=lambda m: None,
    )
    assert gauge["max"] <= 2


def test_concurrent_reconcile_receives_dispatch_order_not_completion_order(cw, tmp_path):
    def fake_run(task, spec=None, repo=None, sandbox_kind=None):
        if task.task == "a":
            time.sleep(0.1)  # a finishes LAST
        return _done_at(task.task)

    run_wave(
        _plan("a", "b"),
        tmp_path,
        run_leaf=fake_run,
        find=_task,
        concurrency=2,
        log=lambda m: None,
    )
    assert cw.reconcile_landed == [("a", "cid-a"), ("b", "cid-b")]


def test_concurrent_worker_crash_fails_only_that_leaf(cw, tmp_path):
    def fake_run(task, spec=None, repo=None, sandbox_kind=None):
        if task.task == "b":
            raise RuntimeError("thread died")
        return _done_at(task.task)

    outcomes = run_wave(
        _plan("a", "b", "c"),
        tmp_path,
        run_leaf=fake_run,
        find=_task,
        concurrency=3,
        log=lambda m: None,
    )
    assert [o.status for o in outcomes] == ["done", "failed", "done"]
    assert "worker crashed" in outcomes[1].reason
    # the crashed leaf never reaches the barrier; batch-mates do
    assert cw.reconcile_landed == [("a", "cid-a"), ("c", "cid-c")]
    assert len(cw.forgotten) == 3  # cleanup includes the crashed leaf's workspace


def test_concurrent_demotion_replaces_outcome_and_journals_own_event(cw, tmp_path):
    import json as _json

    def scripted(landed, on_demote):
        out = []
        for o, _ in landed:
            if o.leaf == "b":
                on_demote("b", "conflict with a")
                out.append(o.model_copy(update={"status": "failed", "reason": "conflict"}))
            else:
                out.append(o)
        return out

    cw.reconcile_result = scripted
    journal_dir = tmp_path / "runs"
    journal_dir.mkdir()
    outcomes = run_wave(
        _plan("a", "b"),
        tmp_path,
        journal_dir=journal_dir,
        run_leaf=lambda t, spec=None, repo=None, sandbox_kind=None: _done_at(t.task),
        find=_task,
        concurrency=2,
        log=lambda m: None,
    )
    assert [o.status for o in outcomes] == ["done", "failed"]
    assert ("b", "Ready") in cw.demotes  # flipped back for replan
    records = [
        _json.loads(line) for line in (journal_dir / "journal.jsonl").read_text().splitlines()
    ]
    dispatches = [r for r in records if r["event"] == "leaf_dispatch" and r["leaf"] == "b"]
    demotions = [r for r in records if r["event"] == "reconcile_demotion" and r["leaf"] == "b"]
    assert len(dispatches) == 1  # ONE dispatch = ONE attempt, however it ends
    assert len(demotions) == 1  # the demotion is its own event, not a second attempt


def test_concurrent_missing_task_gets_no_workspace(cw, tmp_path):
    ran: list[str] = []
    outcomes = run_wave(
        _plan("ghost", "real"),
        tmp_path,
        run_leaf=lambda t, spec=None, repo=None, sandbox_kind=None: (
            ran.append(t.task) or _done_at(t.task)
        ),
        find=lambda title: None if title == "ghost" else _task(title),
        concurrency=2,
        log=lambda m: None,
    )
    assert outcomes[0].status == "skipped"
    assert outcomes[1].status == "done"
    assert ran == ["real"]
    assert len(cw.created) == 1


def test_concurrent_workspace_setup_failure_aborts_with_cleanup(cw, tmp_path, monkeypatch):
    import agents.shared.workspaces as wsmod

    calls = {"n": 0}

    def failing_create(repo, dest, *, base_rev):
        calls["n"] += 1
        if calls["n"] == 2:
            raise wsmod.JJError("workspace add exploded")
        cw.created.append((dest, base_rev))

    monkeypatch.setattr(wsmod, "create_workspace", failing_create)
    ran: list[str] = []
    with pytest.raises(DispatchError, match="workspace setup failed"):
        run_wave(
            _plan("a", "b"),
            tmp_path,
            run_leaf=lambda t, **kw: ran.append(t.task) or _done_at(t.task),
            find=_task,
            concurrency=2,
            log=lambda m: None,
        )
    assert ran == []  # nothing dispatched
    assert len(cw.forgotten) == 1  # the one workspace that DID get created is cleaned up


def test_concurrent_reconcile_error_is_a_loud_dispatch_error(cw, tmp_path, monkeypatch):
    import agents.coding_pipeline.reconcile as rcmod

    def exploding_reconcile(repo, base, landed, *, on_demote, log):
        raise rcmod.ReconcileError("could not reposition")

    monkeypatch.setattr(rcmod, "reconcile_wave", exploding_reconcile)
    with pytest.raises(DispatchError, match="reconcile barrier"):
        run_wave(
            _plan("a"),
            tmp_path,
            run_leaf=lambda t, **kw: _done_at(t.task),
            find=_task,
            concurrency=2,
            log=lambda m: None,
        )
    assert len(cw.forgotten) == 1  # cleanup still ran


def test_serial_path_never_touches_workspace_or_reconcile_code(wired, monkeypatch):
    """The serial-fallback invariant: concurrency 1 is byte-for-byte the old path — no
    workspaces, no reconcile, no scheduler, dx-kind preflight. Pinned explicitly via the
    setting (the shipped default is 3 since the smoke passed) to prove the settings-driven
    path selects serial, not just the explicit argument."""

    def bomb(*a, **kw):
        raise AssertionError("workspace/reconcile code reached from the serial path")

    import agents.coding_pipeline.reconcile as rcmod
    import agents.coding_pipeline.scheduling as schedmod
    import agents.shared.workspaces as wsmod

    monkeypatch.setattr(wsmod, "create_workspace", bomb)
    monkeypatch.setattr(wsmod, "resolve_base_rev", bomb)
    monkeypatch.setattr(rcmod, "reconcile_wave", bomb)
    monkeypatch.setattr(schedmod, "pick_disjoint", bomb)
    monkeypatch.setattr(dp, "_run_concurrent", bomb)
    monkeypatch.setattr(dp.settings, "dispatch_concurrency", 1)

    outcomes = run_wave(
        _plan("a", "b"),
        wired,
        run_leaf=lambda t: _done(t.task),
        find=_task,
        log=lambda m: None,  # concurrency omitted -> settings value (pinned to 1 here)
    )
    assert [o.status for o in outcomes] == ["done", "done"]


def test_concurrent_scheduler_defers_overlapping_scopes(cw, tmp_path):
    """Leaves with colliding predicted scopes never co-dispatch: the later one stays
    Ready for the next wave, and the deferral is journaled — no silent capping."""
    import json as _json

    from agents.coding_pipeline.architect import persist_tree
    from agents.coding_pipeline.models import LeafSpec

    journal_dir = tmp_path / "runs"
    journal_dir.mkdir()
    persist_tree(
        [
            LeafSpec(
                title="a", content="x", feature="F", file_scope=["agents/x.py"], priority=1
            ),
            LeafSpec(
                title="b", content="x", feature="F", file_scope=["agents/x.py"], priority=2
            ),
            LeafSpec(
                title="c", content="x", feature="F", file_scope=["agents/y.py"], priority=3
            ),
        ],
        journal_dir,
    )
    ran: list[str] = []
    outcomes = run_wave(
        _plan("a", "b", "c"),
        tmp_path,
        journal_dir=journal_dir,
        run_leaf=lambda t, spec=None, repo=None, sandbox_kind=None: (
            ran.append(t.task) or _done_at(t.task)
        ),
        find=_task,
        concurrency=3,
        log=lambda m: None,
    )
    assert sorted(ran) == ["a", "c"]  # b's scope collides with a's — deferred
    assert [o.leaf for o in outcomes] == ["a", "c"]
    assert len(cw.created) == 2  # no workspace burned on the deferred leaf
    records = [
        _json.loads(line) for line in (journal_dir / "journal.jsonl").read_text().splitlines()
    ]
    deferred = [r for r in records if r["event"] == "scheduler_deferred"]
    assert deferred and deferred[0]["leaves"] == ["b"]


def test_concurrent_without_any_scope_data_stays_fully_optimistic(cw, tmp_path):
    """No tree.json = no scope data = the picker never engages: the whole ready-set fans
    out and the reconcile barrier is the correctness floor. Prediction reduces wasted
    work when present; it must never become a gate when absent."""
    journal_dir = tmp_path / "runs"
    journal_dir.mkdir()
    ran: list[str] = []
    run_wave(
        _plan("a", "b"),
        tmp_path,
        journal_dir=journal_dir,
        run_leaf=lambda t, spec=None, repo=None, sandbox_kind=None: (
            ran.append(t.task) or _done_at(t.task)
        ),
        find=_task,
        concurrency=2,
        log=lambda m: None,
    )
    assert sorted(ran) == ["a", "b"]


def test_concurrent_scopeless_fixup_serializes_when_tree_has_scopes(cw, tmp_path):
    """Once ANY leaf carries a scope, an unscoped leaf (a replan fix-up) is unknown
    territory: it dispatches alone, never concurrently with a sibling."""
    from agents.coding_pipeline.architect import persist_tree
    from agents.coding_pipeline.models import LeafSpec

    journal_dir = tmp_path / "runs"
    journal_dir.mkdir()
    persist_tree(
        [LeafSpec(title="a", content="x", feature="F", file_scope=["agents/x.py"], priority=1)],
        journal_dir,
    )
    ran: list[str] = []
    run_wave(
        _plan("a", "fixup-not-in-tree"),
        tmp_path,
        journal_dir=journal_dir,
        run_leaf=lambda t, spec=None, repo=None, sandbox_kind=None: (
            ran.append(t.task) or _done_at(t.task)
        ),
        find=_task,
        concurrency=2,
        log=lambda m: None,
    )
    assert ran == ["a"]  # the fix-up waits for the next wave
