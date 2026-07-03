"""Wave planner tests — faked task rows, no Nous, no LLM."""

from __future__ import annotations

from agents.coding_pipeline.models import LeafRow
from agents.coding_pipeline.waves import _row_from_raw, plan_wave


def _row(task: str, **overrides) -> LeafRow:
    base = dict(task=task, status="Ready", execution_mode="Auto-OK", priority=3)
    base.update(overrides)
    return LeafRow.model_validate(base)


def _plan(rows, wave_size=4):
    return plan_wave("Coding Pipeline", "Meta", wave_size=wave_size, rows=rows)


# --- dispatch ordering + cap -----------------------------------------------------


def test_ordering_preferred_then_priority_then_title():
    rows = [
        _row("b-ok-p2", priority=2),
        _row("a-ok-p2", priority=2),
        _row("ok-p1", priority=1),
        _row("preferred-p5", execution_mode="Auto-Preferred", priority=5),
    ]
    plan = _plan(rows)
    assert plan.dispatch == ["preferred-p5", "ok-p1", "a-ok-p2", "b-ok-p2"]


def test_wave_size_caps_dispatch():
    rows = [_row(f"leaf-{i}", priority=i) for i in range(6)]
    plan = _plan(rows, wave_size=2)
    assert plan.dispatch == ["leaf-0", "leaf-1"]
    # capped leaves simply stay Ready in Forge for the next wave — nothing is lost


def test_manual_and_spec_needed_never_dispatch():
    rows = [
        _row("manual", execution_mode="Manual"),
        _row("half-baked", status="Spec Needed", execution_mode="Auto-OK"),
    ]
    plan = _plan(rows)
    assert plan.dispatch == []
    assert plan.ready_manual == 1
    assert plan.spec_needed == 1


def test_blocked_leaves_excluded_and_reported_with_blockers():
    rows = [
        _row("free"),
        _row("stuck", blocked=True, blocked_by=["free", "other"]),
    ]
    plan = _plan(rows)
    assert plan.dispatch == ["free"]
    assert len(plan.blocked) == 1
    assert plan.blocked[0].task == "stuck"
    assert plan.blocked[0].blocked_by == ["free", "other"]


# --- dry vs waiting-on-human -------------------------------------------------------


def test_all_done_is_dry():
    rows = [_row("a", status="Done"), _row("b", status="Done")]
    plan = _plan(rows)
    assert plan.done == 2
    assert plan.dry
    assert not plan.waiting_on_human


def test_manual_outstanding_is_waiting_not_dry():
    rows = [_row("done", status="Done"), _row("manual", execution_mode="Manual")]
    plan = _plan(rows)
    assert not plan.dry
    assert plan.waiting_on_human


def test_blocked_only_is_waiting_not_dry():
    rows = [_row("stuck", blocked=True, blocked_by=["ghost"])]
    plan = _plan(rows)
    assert not plan.dry
    assert plan.waiting_on_human


def test_in_progress_is_waiting_not_dry():
    rows = [_row("running", status="In Progress")]
    plan = _plan(rows)
    assert plan.in_progress == 1
    assert not plan.dry
    assert plan.waiting_on_human


def test_dispatchable_leaves_mean_neither_dry_nor_waiting():
    plan = _plan([_row("go")])
    assert plan.dispatch == ["go"]
    assert not plan.dry
    assert not plan.waiting_on_human


# --- raw-row normalization -----------------------------------------------------------


def test_row_from_raw_normalizes_null_mode_and_priority():
    row = _row_from_raw({"task": "t", "status": "Ready", "execution_mode": None}, False, [])
    assert row.execution_mode == "Manual"  # null-as-manual
    assert row.priority == 99  # missing priority sorts last

    bad_priority = _row_from_raw(
        {"task": "t2", "status": "Ready", "priority": "high"}, True, ["dep"]
    )
    assert bad_priority.priority == 99
    assert bad_priority.blocked and bad_priority.blocked_by == ["dep"]
