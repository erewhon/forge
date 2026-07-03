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
from agents.coding_pipeline.models import FramingProposal, GoalSpec, Inventory


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
