"""Inventory collector tests over a tmp fixture repo — pure filesystem, no LLM, no Nous."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.coding_pipeline import inventory as inv_mod
from forge.coding_pipeline.inventory import (
    collect_inventory,
    goal_terms,
    render_inventory,
    run_dir_for,
    write_inventory,
)
from forge.coding_pipeline.models import ExistingTask, GoalSpec


@pytest.fixture
def repo(tmp_path) -> Path:
    """A small fixture project: python package, tests, key files, ignored + gitignored dirs."""
    root = tmp_path / "fixture-proj"
    (root / "agents" / "exporter").mkdir(parents=True)
    (root / "agents" / "exporter" / "__init__.py").write_text("")
    (root / "agents" / "exporter" / "json_export.py").write_text("def export(): ...\n")
    (root / "agents" / "exporter" / "deep" / "deeper").mkdir(parents=True)
    (root / "agents" / "exporter" / "deep" / "deeper" / "buried.py").write_text("")
    (root / "tests").mkdir()
    (root / "tests" / "test_export.py").write_text("def test_x(): ...\n")
    (root / "CLAUDE.md").write_text("# Conventions\nuse uv\n")
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    (root / "uv.lock").write_text("")
    (root / "node_modules" / "junk").mkdir(parents=True)
    (root / "node_modules" / "junk" / "x.js").write_text("")
    (root / "secretstuff").mkdir()
    (root / "secretstuff" / "hidden.py").write_text("")
    (root / ".gitignore").write_text("secretstuff/\n*.tmp\n")
    return root


def _goal(**overrides) -> GoalSpec:
    base = dict(
        goal="Add a JSON export command",
        project="Fixture",
        epic_slug="json-export",
    )
    base.update(overrides)
    return GoalSpec.model_validate(base)


def test_tree_is_ignore_aware_and_depth_capped(repo):
    inv = collect_inventory(_goal(), repo, existing_tasks=[])
    assert "node_modules" not in inv.tree  # default ignore
    assert "secretstuff" not in inv.tree  # simple .gitignore entry honored
    assert "json_export.py" in inv.tree  # depth 3 reached
    assert "buried.py" not in inv.tree  # depth 4 capped


def test_key_files_toolchain_modules_and_tests(repo):
    inv = collect_inventory(_goal(), repo, existing_tasks=[])
    assert [k.path for k in inv.key_files] == ["CLAUDE.md", "pyproject.toml"]
    assert "use uv" in inv.key_files[0].head
    assert "uv/python" in inv.toolchain
    assert "python (pyproject)" in inv.toolchain
    assert inv.modules == ["agents", "tests"]
    assert "tests" in inv.test_layout


def test_overlap_scan_matches_goal_terms_against_paths(repo):
    inv = collect_inventory(_goal(goal="Add a JSON export command"), repo, existing_tasks=[])
    assert any("json_export.py" in p for p in inv.overlaps)
    # and terms exclude stopwords/short words
    terms = goal_terms(_goal(goal="Build this with json export"))
    assert "json" in terms and "export" in terms
    assert "this" not in terms and "with" not in terms


def test_existing_tasks_capped_and_counted(repo, monkeypatch):
    monkeypatch.setattr(inv_mod, "EXISTING_TASKS_MAX", 2)
    tasks = [ExistingTask(task=f"t{i}", status="Ready") for i in range(5)]
    inv = collect_inventory(_goal(), repo, existing_tasks=tasks)
    assert len(inv.existing_tasks) == 2
    assert inv.truncated >= 3  # dropped tasks are counted, not silent


def test_tree_entry_cap_counts_drops(repo, monkeypatch):
    monkeypatch.setattr(inv_mod, "TREE_MAX_ENTRIES", 3)
    inv = collect_inventory(_goal(), repo, existing_tasks=[])
    assert len(inv.tree.splitlines()) == 3
    assert inv.truncated > 0


def test_render_and_write_inventory(repo, tmp_path):
    tasks = [ExistingTask(task="Old task", status="Done", external_ref="pipeline:x:y")]
    inv = collect_inventory(_goal(), repo, existing_tasks=tasks)
    doc = render_inventory(inv)
    assert "# Inventory — Fixture" in doc
    assert "Old task — Done [pipeline:x:y]" in doc
    assert "CLAUDE.md (head)" in doc

    run_dir = tmp_path / "runs" / "json-export"
    md = write_inventory(inv, run_dir)
    assert md.read_text() == doc
    assert (run_dir / "inventory.json").exists()


def test_render_trims_to_config_cap(repo, monkeypatch):
    from forge.coding_pipeline.config import settings

    monkeypatch.setattr(settings, "inventory_max_chars", 500)
    inv = collect_inventory(_goal(), repo, existing_tasks=[])
    doc = render_inventory(inv)
    assert "trimmed to fit the cap" in doc
    assert len(doc) < 700  # cap + trim notice


def test_run_dir_uses_epic_slug_or_derives(monkeypatch, tmp_path):
    from forge.coding_pipeline.config import settings

    monkeypatch.setattr(settings, "runs_dir", tmp_path)
    assert run_dir_for(_goal()) == tmp_path / "json-export"
    assert run_dir_for(_goal(epic_slug=None)) == tmp_path / "add-a-json-export-command"
