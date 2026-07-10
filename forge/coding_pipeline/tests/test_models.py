"""Model tests: loading, validation guards, and journal round-trips (no LLM, no IO beyond tmp)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from forge.coding_pipeline.models import (
    BoundednessCheck,
    EscalateAction,
    FixupAction,
    FramingProposal,
    GoalSpec,
    HaltAction,
    LeafOutcome,
    LeafSpec,
    SuiteResult,
    TaskTree,
    WaveRecord,
    WaveReport,
)

EXAMPLES = Path(__file__).parent.parent / "examples"


def _leaf(title: str = "wire the flux capacitor") -> LeafSpec:
    return LeafSpec(title=title, content="spec body", feature="Time Travel")


# --- GoalSpec loading --------------------------------------------------------


def test_goal_spec_loads_example_yaml():
    spec = GoalSpec.load(EXAMPLES / "sample-goal.yaml")
    assert spec.project == "Notes"
    assert spec.epic_slug == "notes-json-export"
    assert spec.value_hints


def test_goal_spec_loads_yaml(tmp_path):
    p = tmp_path / "goal.yaml"
    p.write_text("goal: build X\nproject: Meta\n")
    spec = GoalSpec.load(p)
    assert spec.goal == "build X"
    assert spec.project == "Meta"
    assert spec.context == ""


def test_goal_spec_loads_markdown_frontmatter_with_body_as_context(tmp_path):
    p = tmp_path / "goal.md"
    p.write_text("---\ngoal: build X\nproject: Meta\ncontext: existing note\n---\n\nBody detail.\n")
    spec = GoalSpec.load(p)
    assert spec.goal == "build X"
    assert spec.context == "existing note\n\nBody detail."


def test_goal_spec_markdown_requires_frontmatter(tmp_path):
    p = tmp_path / "goal.md"
    p.write_text("just prose, no frontmatter")
    with pytest.raises(ValueError, match="frontmatter"):
        GoalSpec.load(p)


def test_goal_spec_rejects_unknown_suffix(tmp_path):
    p = tmp_path / "goal.txt"
    p.write_text("goal: x\nproject: y\n")
    with pytest.raises(ValueError, match="unsupported"):
        GoalSpec.load(p)


# --- FramingProposal ---------------------------------------------------------


def _framing(**overrides) -> FramingProposal:
    base = dict(
        goal_as_stated="build web parity",
        restated_goal="serve the desktop frontend from the daemon",
        rescoped=True,
        recommendation="platform-shim approach",
        epic_slug="web-shim",
    )
    base.update(overrides)
    return FramingProposal.model_validate(base)


def test_framing_defaults_unapproved():
    assert _framing().approved is False


def test_framing_rejects_unsafe_slug():
    with pytest.raises(ValidationError, match="slug"):
        _framing(epic_slug="Web Shim!")


def test_framing_round_trips_json():
    f = _framing(risks=["scope creep"], value_ordering=["read-only view first"])
    revived = FramingProposal.model_validate_json(f.model_dump_json())
    assert revived == f


# --- LeafSpec guards ---------------------------------------------------------


def test_leaf_defaults_are_conservative():
    leaf = _leaf()
    assert leaf.execution_mode == "Manual"
    assert leaf.status == "Ready"
    assert leaf.requires_tests is True


def test_leaf_title_with_comma_rejected():
    with pytest.raises(ValidationError, match="comma"):
        _leaf("scaffold, models, and config")


def test_leaf_dependency_with_comma_rejected():
    with pytest.raises(ValidationError, match="comma"):
        LeafSpec(
            title="ok title",
            content="body",
            feature="F",
            depends_on=["scaffold, models, and config"],
        )


def test_boundedness_worker_shaped_requires_all_five():
    all_good = BoundednessCheck(
        single_concern=True,
        bounded_diff=True,
        small_estimate=True,
        testable_acceptance=True,
        files_named=True,
    )
    assert all_good.worker_shaped
    one_bad = all_good.model_copy(update={"files_named": False})
    assert not one_bad.worker_shaped


# --- wave report + record round-trips -----------------------------------------


def test_wave_report_partitions_outcomes_and_gates_on_suite():
    report = WaveReport(
        wave=1,
        outcomes=[
            LeafOutcome(leaf="a", status="done", commit_id="abc"),
            LeafOutcome(leaf="b", status="failed", reason="tests red"),
            LeafOutcome(leaf="c", status="skipped", reason="gate refused"),
        ],
        suite=SuiteResult(passed=False, output_tail="boom"),
    )
    assert [o.leaf for o in report.landed] == ["a"]
    assert [o.leaf for o in report.failed] == ["b"]
    assert report.suite_green is False


def test_wave_report_suite_green_requires_a_suite_run():
    assert WaveReport(wave=1).suite_green is False


def test_wave_record_round_trips_discriminated_actions():
    record = WaveRecord(
        wave=2,
        dispatched=["a", "b"],
        report=WaveReport(wave=2, suite=SuiteResult(passed=True)),
        actions=[
            FixupAction(finding_slug="dangling-ref", leaf=_leaf("fix dangling ref")),
            EscalateAction(leaf_title="b", diagnostics="failed twice"),
            HaltAction(reason="framing invalidated"),
        ],
    )
    revived = WaveRecord.model_validate_json(record.model_dump_json())
    assert revived == record
    assert [a.kind for a in revived.actions] == ["fixup", "escalate", "halt"]
    assert isinstance(revived.actions[0], FixupAction)  # union revives the right class
    assert revived.actions[0].leaf.title == "fix dangling ref"


# --- LeafSpec file_scope -----------------------------------------------------


def test_leaf_file_scope_defaults_empty():
    leaf = _leaf()
    assert leaf.file_scope == []


def test_leaf_file_scope_accepted_values():
    leaf = LeafSpec(
        title="scaffold models",
        content="spec",
        feature="F",
        file_scope=["forge/coding_pipeline/models.py", "forge/shared/"],
    )
    assert leaf.file_scope == ["forge/coding_pipeline/models.py", "forge/shared/"]


def test_leaf_file_scope_comma_rejected():
    with pytest.raises(ValidationError, match="comma"):
        LeafSpec(
            title="ok title",
            content="body",
            feature="F",
            file_scope=["forge/foo.py,baz.py"],
        )


def test_file_scope_round_trips_through_json():
    leaf = LeafSpec(
        title="scope leaf",
        content="spec",
        feature="F",
        file_scope=["path/to/file.py", "dir/"],
    )
    revived = LeafSpec.model_validate_json(leaf.model_dump_json())
    assert revived.file_scope == ["path/to/file.py", "dir/"]


def test_legacy_tree_json_without_file_scope_loads_empty(tmp_path):
    """Old tree.json files that lack file_scope must load with []."""
    import json

    legacy = '{"leaves": [{"title": "old leaf", "content": "spec", "feature": "F"}]}'
    p = tmp_path / "tree.json"
    p.write_text(legacy)
    tree = TaskTree.model_validate(json.loads(legacy))
    assert tree.leaves[0].file_scope == []


def test_task_tree_round_trip_with_file_scope(tmp_path):
    leaves = [
        LeafSpec(
            title="scoped leaf",
            content="spec",
            feature="F",
            file_scope=["a/b.py"],
        ),
        LeafSpec(title="unscoped leaf", content="spec", feature="F"),
    ]
    tree = TaskTree(leaves=leaves)
    json_str = tree.model_dump_json(indent=2)
    revived = TaskTree.model_validate_json(json_str)
    assert revived.leaves[0].file_scope == ["a/b.py"]
    assert revived.leaves[1].file_scope == []
