"""Framing-stage tests: mocked pool (no LLM), persistence guards, and the approval gate."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agents.coding_pipeline import architect as arch
from agents.coding_pipeline.architect import (
    ArchitectError,
    FramingExistsError,
    FramingNotApprovedError,
    approve_framing,
    load_framing,
    persist_framing,
    propose_framing,
    render_framing,
    require_approved_framing,
)
from agents.coding_pipeline.models import (
    FramingProposal,
    GoalSpec,
    Inventory,
    LeafSpec,
    TaskTree,
)


def _proposal(**overrides) -> FramingProposal:
    base = dict(
        goal_as_stated="build web parity",
        restated_goal="serve the desktop frontend from the daemon",
        rescoped=True,
        recommendation="platform-shim approach",
        epic_slug="web-shim",
        risks=["scope creep"],
        value_ordering=["read-only view first"],
    )
    base.update(overrides)
    return FramingProposal.model_validate(base)


def _goal(**overrides) -> GoalSpec:
    base = dict(goal="build web parity", project="Nous")
    base.update(overrides)
    return GoalSpec.model_validate(base)


def _inventory() -> Inventory:
    return Inventory(project="Nous", repo="/repo", tree="src/\n  app.rs")


def _mock_structured(monkeypatch, value, error=None):
    calls: list[dict] = []

    def fake(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(value=value, error=error, ok=value is not None)

    monkeypatch.setattr(arch, "structured", fake)
    return calls


# --- propose_framing -----------------------------------------------------------


def test_propose_framing_forces_approved_false(monkeypatch):
    # even a model that claims approval gets it stripped
    _mock_structured(monkeypatch, _proposal(approved=True))
    out = propose_framing(_goal(), _inventory())
    assert out.approved is False


def test_propose_framing_prompt_carries_goal_and_inventory(monkeypatch):
    calls = _mock_structured(monkeypatch, _proposal())
    propose_framing(_goal(context="daemon exists", value_hints=["viewer first"]), _inventory())
    user = calls[0]["user"]
    assert "build web parity" in user
    assert "daemon exists" in user
    assert "- viewer first" in user
    assert "app.rs" in user  # rendered inventory included
    assert calls[0]["schema"] is FramingProposal


def test_propose_framing_human_slug_beats_model_slug(monkeypatch):
    _mock_structured(monkeypatch, _proposal(epic_slug="model-idea"))
    out = propose_framing(_goal(epic_slug="human-choice"), _inventory())
    assert out.epic_slug == "human-choice"


def test_propose_framing_pool_exhaustion_raises(monkeypatch):
    _mock_structured(monkeypatch, None, error="pool exhausted")
    with pytest.raises(ArchitectError, match="pool exhausted"):
        propose_framing(_goal(), _inventory())


# --- persistence + approval gate -------------------------------------------------


def test_persist_refuses_overwrite_without_force(tmp_path):
    persist_framing(_proposal(), tmp_path)
    with pytest.raises(FramingExistsError):
        persist_framing(_proposal(recommendation="new idea"), tmp_path)
    # force overwrites
    persist_framing(_proposal(recommendation="new idea"), tmp_path, force=True)
    assert load_framing(tmp_path).recommendation == "new idea"


def test_persist_writes_json_and_md(tmp_path):
    persist_framing(_proposal(), tmp_path)
    assert (tmp_path / "framing.json").exists()
    md = (tmp_path / "framing.md").read_text()
    assert "Architect push-back" in md  # rescoped framing is called out
    assert "NO — review and approve" in md


def test_gate_refuses_missing_and_unapproved_framing(tmp_path):
    with pytest.raises(ArchitectError, match="no framing.json"):
        require_approved_framing(tmp_path)
    persist_framing(_proposal(), tmp_path)
    with pytest.raises(FramingNotApprovedError):
        require_approved_framing(tmp_path)


def test_approve_flips_gate_open(tmp_path):
    persist_framing(_proposal(), tmp_path)
    approved = approve_framing(tmp_path)
    assert approved.approved is True
    # gate now passes, and the persisted md reflects approval
    assert require_approved_framing(tmp_path).approved is True
    assert "Approved:** yes" in (tmp_path / "framing.md").read_text()


def test_approve_without_framing_raises(tmp_path):
    with pytest.raises(ArchitectError, match="no framing.json"):
        approve_framing(tmp_path)


def test_render_framing_unrescoped_has_no_pushback_banner():
    md = render_framing(_proposal(rescoped=False))
    assert "Architect push-back" not in md
    assert "## Restated goal" in md


# --- A2: decomposition ------------------------------------------------------


def _leaf(title: str, **overrides) -> LeafSpec:
    base = dict(title=title, content="spec", feature="Web Shim", estimate="s")
    base.update(overrides)
    return LeafSpec.model_validate(base)


def _shaped(title: str, **flags) -> arch.LeafBoundedness:
    base = dict(
        leaf_title=title,
        single_concern=True,
        bounded_diff=True,
        small_estimate=True,
        testable_acceptance=True,
        files_named=True,
    )
    base.update(flags)
    return arch.LeafBoundedness.model_validate(base)


def _mock_decompose(monkeypatch, leaves, verdicts=None):
    """Wire structured() to return a TaskTree and discover() to return boundedness verdicts.
    verdicts=None means 'all leaves pass'."""
    calls = _mock_structured(monkeypatch, TaskTree(leaves=leaves) if leaves else None)
    if verdicts is None:
        verdicts = [_shaped(leaf.title) for leaf in leaves or []]
    monkeypatch.setattr(arch, "discover", lambda finders, **kw: verdicts)
    return calls


def test_decompose_refuses_unapproved_framing(monkeypatch):
    _mock_decompose(monkeypatch, [_leaf("a")])
    with pytest.raises(FramingNotApprovedError):
        arch.decompose(_proposal(approved=False), _inventory())


def test_decompose_applies_conservative_floor(monkeypatch):
    leaves = [
        _leaf("auto leaf", execution_mode="Auto-OK", requires_tests=False, max_files=None),
        _leaf("novel leaf", execution_mode="Auto-OK", complexity="novel"),
        _leaf("specless leaf", execution_mode="Auto-Preferred", status="Spec Needed"),
        _leaf("blank feature", feature=" "),
    ]
    _mock_decompose(monkeypatch, leaves)
    out = arch.decompose(_proposal(approved=True), _inventory())
    auto = next(leaf for leaf in out if leaf.title == "auto leaf")
    assert auto.requires_tests is True  # auto always tests
    assert auto.max_files == 5  # auto always capped
    assert next(le for le in out if le.title == "novel leaf").execution_mode == "Manual"
    assert next(le for le in out if le.title == "specless leaf").execution_mode == "Manual"
    assert next(le for le in out if le.title == "blank feature").feature == "Web Shim"


def test_decompose_rejects_unknown_dep(monkeypatch):
    _mock_decompose(monkeypatch, [_leaf("a", depends_on=["ghost"])])
    with pytest.raises(ArchitectError, match="unknown titles"):
        arch.decompose(_proposal(approved=True), _inventory())


def test_decompose_rejects_dependency_cycle(monkeypatch):
    _mock_decompose(monkeypatch, [_leaf("a", depends_on=["b"]), _leaf("b", depends_on=["a"])])
    with pytest.raises(ArchitectError, match="cycle"):
        arch.decompose(_proposal(approved=True), _inventory())


def test_boundedness_failure_demotes_auto_leaf(monkeypatch):
    leaves = [_leaf("wobbly", execution_mode="Auto-OK")]
    _mock_decompose(
        monkeypatch, leaves, verdicts=[_shaped("wobbly", files_named=False, notes="no files")]
    )
    out = arch.decompose(_proposal(approved=True), _inventory())
    assert out[0].execution_mode == "Manual"
    assert out[0].complexity == "novel"
    assert out[0].boundedness is not None and not out[0].boundedness.worker_shaped


def test_missing_boundedness_verdict_demotes_fail_closed(monkeypatch):
    leaves = [_leaf("unchecked", execution_mode="Auto-OK")]
    _mock_decompose(monkeypatch, leaves, verdicts=[])  # finder dropped, no verdict
    out = arch.decompose(_proposal(approved=True), _inventory())
    assert out[0].execution_mode == "Manual"
    assert "unavailable" in out[0].boundedness.notes


def test_manual_leaf_keeps_mode_despite_failing_boundedness(monkeypatch):
    leaves = [_leaf("big design", execution_mode="Manual", complexity="novel")]
    _mock_decompose(monkeypatch, leaves, verdicts=[_shaped("big design", single_concern=False)])
    out = arch.decompose(_proposal(approved=True), _inventory())
    assert out[0].execution_mode == "Manual"  # terminal state, not an error


def test_decompose_pool_exhaustion_raises(monkeypatch):
    _mock_decompose(monkeypatch, None)
    with pytest.raises(ArchitectError, match="no usable tree"):
        arch.decompose(_proposal(approved=True), _inventory())


def test_persist_and_load_tree_round_trip(tmp_path, monkeypatch):
    leaves = [_leaf("a"), _leaf("b", depends_on=["a"])]
    arch.persist_tree(leaves, tmp_path)
    assert arch.load_tree(tmp_path) == leaves
    md = (tmp_path / "tree.md").read_text()
    assert "**a**" in md and "← a" in md
