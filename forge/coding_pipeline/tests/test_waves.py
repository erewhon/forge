"""Wave planner tests — faked task rows, no Nous, no LLM."""

from __future__ import annotations

from forge.coding_pipeline.models import LeafRow
from forge.coding_pipeline.waves import _row_from_raw, epic_ref_prefix, is_epic_row, plan_wave


def _row(task: str, **overrides) -> LeafRow:
    base = dict(task=task, status="Ready", execution_mode="Auto-OK", priority=3)
    base.update(overrides)
    return LeafRow.model_validate(base)


def _plan(rows, wave_size=4):
    return plan_wave("toy-epic", "Meta", wave_size=wave_size, rows=rows)


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


# --- epic membership (ref-prefix scoping) ------------------------------------------
# Regression for the e2e dry-run finding: decomposition split one epic across three
# Feature values; feature-scoped planning hid real leaves from dispatch and from the
# dry/exhausted logic. Membership is the pipeline's own ref prefix, whatever the
# Feature column says.


def test_is_epic_row_matches_on_ref_prefix_across_features():
    tree_leaf = {"external_ref": "pipeline:toy-epic:temperature-domain-impl", "feature": "A"}
    other_feature = {"external_ref": "pipeline:toy-epic:list-units-add", "feature": "B"}
    fixup = {"external_ref": "pipeline:toy-epic:fix:off-by-one", "feature": "C"}
    assert is_epic_row(tree_leaf, "toy-epic")
    assert is_epic_row(other_feature, "toy-epic")
    assert is_epic_row(fixup, "toy-epic")  # replan fix-ups are epic members too


def test_is_epic_row_rejects_other_epics_and_unreffed_tasks():
    assert not is_epic_row({"external_ref": "pipeline:other-epic:leaf"}, "toy-epic")
    assert not is_epic_row({"external_ref": ""}, "toy-epic")
    assert not is_epic_row({}, "toy-epic")
    # a slug that merely PREFIXES another must not match (toy-epic vs toy-epic-2)
    assert not is_epic_row({"external_ref": "pipeline:toy-epic-2:leaf"}, "toy-epic")


def test_is_epic_row_feature_narrowing():
    row = {"external_ref": "pipeline:toy-epic:x", "feature": "Temperature Domain"}
    assert is_epic_row(row, "toy-epic", feature="Temperature Domain")
    assert not is_epic_row(row, "toy-epic", feature="List Units Command")


def test_epic_ref_prefix_shape():
    assert epic_ref_prefix("toy-epic") == "pipeline:toy-epic:"


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


def test_journal_landed_overrides_a_rearmed_ready_row():
    # A leaf the journal says landed must never redispatch, whatever Forge claims —
    # a replan or a human re-arming a finished task re-runs merged work otherwise
    # (deps-v2 wave 11, live).
    rows = [_row("landed-leaf"), _row("fresh-leaf")]
    plan = plan_wave("toy-epic", "Meta", wave_size=4, rows=rows, landed_titles={"landed-leaf"})
    assert plan.dispatch == ["fresh-leaf"]
    assert plan.done == 1
