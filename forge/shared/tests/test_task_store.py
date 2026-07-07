"""Task-store port tests — the factory selects a backend, and ForgeTaskStore forwards
every operation to the Nous-backed implementation unchanged.

Delegation is verified by patching each *source* function (``nous_client.*`` /
``forge_emit.emit_tasks`` / the ``waves`` normalizer) and asserting ForgeTaskStore
forwards its arguments and returns the result verbatim. No Nous daemon is touched.
"""

from __future__ import annotations

import pytest

from agents.shared import task_store
from agents.shared.task_store import ForgeTaskStore, TaskStore, get_task_store


def test_factory_defaults_to_forge(monkeypatch):
    monkeypatch.setattr(task_store.settings, "backend", "forge")
    store = get_task_store()
    assert isinstance(store, ForgeTaskStore)
    assert isinstance(store, TaskStore)  # runtime_checkable Protocol conformance


def test_factory_rejects_unknown_backend(monkeypatch):
    monkeypatch.setattr(task_store.settings, "backend", "sqlite")
    with pytest.raises(ValueError, match="unknown TASK_STORE_BACKEND"):
        get_task_store()


def test_factory_backend_is_case_and_space_insensitive(monkeypatch):
    monkeypatch.setattr(task_store.settings, "backend", "  Forge  ")
    assert isinstance(get_task_store(), ForgeTaskStore)


def test_update_status_forwards_to_nous_client(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "agents.task_worker.nous_client.update_task_status",
        lambda task, status, notes="", execution_mode=None: seen.update(
            task=task, status=status, notes=notes, mode=execution_mode
        ),
    )
    ForgeTaskStore().update_status("leaf-a", "Ready", notes="n", execution_mode="Manual")
    assert seen == {"task": "leaf-a", "status": "Ready", "notes": "n", "mode": "Manual"}


def test_find_task_forwards_and_returns(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        "agents.task_worker.nous_client.find_task",
        lambda name: sentinel if name == "leaf-a" else None,
    )
    assert ForgeTaskStore().find_task("leaf-a") is sentinel
    assert ForgeTaskStore().find_task("missing") is None


def test_next_ready_forwards_projects(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "agents.task_worker.nous_client.find_next_task",
        lambda projects: seen.update(projects=projects) or "picked",
    )
    assert ForgeTaskStore().next_ready(["Meta", "Nous"]) == "picked"
    assert seen == {"projects": ["Meta", "Nous"]}


def test_get_spec_and_worker_gate_forward(monkeypatch):
    monkeypatch.setattr("agents.task_worker.nous_client.get_task_spec", lambda name: f"SPEC:{name}")
    monkeypatch.setattr(
        "agents.task_worker.nous_client.check_worker_gate",
        lambda name: "" if name == "ok" else "blocked",
    )
    store = ForgeTaskStore()
    assert store.get_spec("leaf-a") == "SPEC:leaf-a"
    assert store.worker_gate("ok") == ""
    assert store.worker_gate("nope") == "blocked"


def test_emit_forwards_batch_and_gating(monkeypatch):
    seen = {}

    def fake_emit_tasks(specs, **kwargs):
        seen["specs"] = specs
        seen.update(kwargs)
        return "SUMMARY"

    monkeypatch.setattr("agents.shared.forge_emit.emit_tasks", fake_emit_tasks)
    result = ForgeTaskStore().emit(
        ["s1", "s2"], project="Meta", status="Ready", dry_run=True, max_per_run=5
    )
    assert result == "SUMMARY"
    assert seen["specs"] == ["s1", "s2"]
    assert seen["project"] == "Meta"
    assert seen["status"] == "Ready"
    assert seen["dry_run"] is True
    assert seen["max_per_run"] == 5


def test_list_rows_queries_project_and_normalizes(monkeypatch):
    seen = {}

    monkeypatch.setattr(
        "agents.task_worker.nous_client._read_db_content", lambda: {"db": "content"}
    )

    def fake_query(db_content, **kwargs):
        seen["db"] = db_content
        seen.update(kwargs)
        return [{"task": "raw-1"}]

    monkeypatch.setattr("nous_mcp.workflow._query_tasks", fake_query)
    monkeypatch.setattr(
        "agents.coding_pipeline.waves._rows_from_raw",
        lambda raw, db: [f"row:{r['task']}" for r in raw],
    )

    rows = ForgeTaskStore().list_rows("Meta", feature="Temp Domain", include_done=False)
    assert rows == ["row:raw-1"]
    assert seen["db"] == {"db": "content"}
    assert seen["project"] == "Meta"
    assert seen["feature"] == "Temp Domain"
    assert seen["include_done"] is False


def test_in_progress_titles_filters_by_ref_prefix(monkeypatch):
    monkeypatch.setattr(
        "agents.task_worker.nous_client._read_db_content", lambda: {"db": "content"}
    )
    monkeypatch.setattr(
        "nous_mcp.workflow._query_tasks",
        lambda db_content, **kwargs: [
            {"task": "in-epic", "external_ref": "pipeline:toy-epic:leaf-a"},
            {"task": "other-epic", "external_ref": "pipeline:other:leaf-b"},
            {"task": "unreffed", "external_ref": ""},
            {"task": "  ", "external_ref": "pipeline:toy-epic:blank"},  # blank title dropped
        ],
    )
    titles = ForgeTaskStore().in_progress_titles("pipeline:toy-epic:")
    assert titles == ["in-epic"]
