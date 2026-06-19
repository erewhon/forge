"""Tests for the shared Forge-task emitter (create_task + existing-refs mocked, no daemon)."""

from __future__ import annotations

import json

import pytest

from agents.shared import forge_emit
from agents.shared.forge_emit import EmitSpec, emit_task, emit_tasks


@pytest.fixture
def captured_creates(monkeypatch):
    """Patch the create_task indirection to record calls instead of hitting the daemon."""
    calls: list[dict] = []

    def fake_create(**kwargs) -> str:
        calls.append(kwargs)
        return json.dumps({"page_id": f"pg-{len(calls)}", "row_id": f"row-{len(calls)}"})

    monkeypatch.setattr(forge_emit, "_create_task", fake_create)
    return calls


def _no_existing(monkeypatch):
    monkeypatch.setattr(forge_emit, "existing_external_refs", lambda: set())


# --- emit_task --------------------------------------------------------------


def test_emit_task_creates_with_gating_defaults(captured_creates, monkeypatch):
    _no_existing(monkeypatch)
    out = emit_task(
        project="Meta",
        title="Refactor foo [duplication]",
        content="body",
        external_ref="refactor:m::foo:duplication",
        task_type="refactor",
        estimate="s",
    )
    assert out.action == "created"
    assert out.detail == "pg-1"
    assert len(captured_creates) == 1
    sent = captured_creates[0]
    # review-then-implement gate: keeps it away from the autonomous worker
    assert sent["status"] == "Spec Needed"
    assert sent["execution_mode"] == "Manual"
    assert sent["external_ref"] == "refactor:m::foo:duplication"
    assert sent["task_type"] == "refactor"
    assert sent["estimate"] == "s"
    assert sent["phase"] == "Polish"


def test_emit_task_skips_existing_ref(captured_creates):
    out = emit_task(
        project="Meta",
        title="dup",
        content="b",
        external_ref="refactor:x:dup",
        existing_refs={"refactor:x:dup"},
    )
    assert out.action == "skipped"
    assert captured_creates == []  # no write


def test_emit_task_dry_run_creates_nothing(captured_creates, monkeypatch):
    _no_existing(monkeypatch)
    out = emit_task(project="Meta", title="t", content="b", external_ref="r1", dry_run=True)
    assert out.action == "dry-run"
    assert captured_creates == []


def test_emit_task_mutates_passed_refs(captured_creates):
    refs: set[str] = set()
    emit_task(project="Meta", title="t", content="b", external_ref="r1", existing_refs=refs)
    assert "r1" in refs  # so a later identical ref dedups within the run


# --- emit_tasks (batch) -----------------------------------------------------


def _spec(ref: str, title: str = "t") -> EmitSpec:
    return EmitSpec(title=title, content="body", external_ref=ref, task_type="refactor")


def test_emit_tasks_dedups_against_existing(captured_creates, monkeypatch):
    monkeypatch.setattr(forge_emit, "existing_external_refs", lambda: {"r-old"})
    summary = emit_tasks(
        [_spec("r-old"), _spec("r-new")],
        project="Meta",
    )
    assert len(summary.created) == 1
    assert len(summary.skipped) == 1
    assert summary.created[0].external_ref == "r-new"
    assert [c["external_ref"] for c in captured_creates] == ["r-new"]


def test_emit_tasks_dedups_within_run(captured_creates, monkeypatch):
    _no_existing(monkeypatch)
    summary = emit_tasks([_spec("dup"), _spec("dup")], project="Meta")
    assert len(summary.created) == 1
    assert len(summary.skipped) == 1
    assert len(captured_creates) == 1


def test_emit_tasks_caps_creations(captured_creates, monkeypatch):
    _no_existing(monkeypatch)
    specs = [_spec(f"r{i}") for i in range(5)]
    summary = emit_tasks(specs, project="Meta", max_per_run=2)
    assert len(summary.created) == 2
    assert summary.capped == 3
    assert len(captured_creates) == 2  # cap is a real write ceiling, not just a count


def test_emit_tasks_cap_not_consumed_by_dedup(captured_creates, monkeypatch):
    # two dedup skips ahead of two fresh specs; cap=2 should still create both fresh ones
    monkeypatch.setattr(forge_emit, "existing_external_refs", lambda: {"old1", "old2"})
    specs = [_spec("old1"), _spec("old2"), _spec("new1"), _spec("new2")]
    summary = emit_tasks(specs, project="Meta", max_per_run=2)
    assert len(summary.created) == 2
    assert summary.capped == 0
    assert len(summary.skipped) == 2


def test_emit_tasks_dry_run_creates_nothing(captured_creates, monkeypatch):
    _no_existing(monkeypatch)
    summary = emit_tasks([_spec("r1"), _spec("r2")], project="Meta", dry_run=True)
    assert len(summary.planned) == 2
    assert len(summary.created) == 0
    assert captured_creates == []
