"""Epic-context preamble tests — the sibling-contracts injection (e2e dry-run Q1).

The extraction path runs against a real jj repo (the drift case was real code:
``VALID_UNITS`` vs an assumed factor table); the composition/truncation logic runs
against fakes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from agents.coding_pipeline import context as ctx
from agents.coding_pipeline.models import LeafRow
from agents.task_worker.models import TaskInfo

# --- fixtures ---------------------------------------------------------------------


def _task(title: str = "Extend CLI router", deps: list[str] | None = None) -> TaskInfo:
    return TaskInfo(
        id="row-1",
        task=title,
        project="Pipeline-Smoke",
        status="Ready",
        priority=2,
        execution_mode="Auto-OK",
        deps=deps or [],
    )


def _row(task: str, status: str = "Ready") -> LeafRow:
    return LeafRow(task=task, status=status, execution_mode="Auto-OK", priority=3)


def _journal(run_dir, *records):
    (run_dir / "journal.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _landed(leaf: str, commit: str) -> dict:
    return {"event": "leaf_dispatch", "leaf": leaf, "status": "done", "commit_id": commit}


# --- journal scan -------------------------------------------------------------------


def test_landed_commits_last_done_wins(tmp_path):
    _journal(
        tmp_path,
        _landed("mod", "aaa"),
        {"event": "leaf_dispatch", "leaf": "mod", "status": "failed", "reason": "x"},
        _landed("mod", "bbb"),  # re-landed after a revert: newest commit wins
        {"event": "gate_result", "gate": "suite", "passed": True},
    )
    assert ctx.landed_commits(tmp_path) == {"mod": "bbb"}


def test_landed_commits_missing_journal_is_empty(tmp_path):
    assert ctx.landed_commits(tmp_path) == {}


# --- public_interface (pure ast) ------------------------------------------------------


def test_public_interface_signatures_constants_and_classes():
    source = '''
"""Module docstring."""
VALID_UNITS: frozenset[str] = frozenset(("C", "F", "K"))
KG_PER_LB = 0.45359237
_PRIVATE = 1

def convert(value: float, from_unit: str, to_unit: str) -> float:
    return value

def _helper():
    pass

class Converter(Base):
    def __init__(self, table: dict):
        pass
    def convert(self, value: float) -> float:
        return value
    def _internal(self):
        pass
'''
    surface = ctx.public_interface(source)
    assert "VALID_UNITS: frozenset[str]" in surface  # the dry-run's actual drift case
    assert "KG_PER_LB" in surface
    assert "def convert(value: float, from_unit: str, to_unit: str) -> float" in surface
    assert "class Converter(Base)" in surface
    assert "    def convert(self, value: float) -> float" in surface
    assert not any("_PRIVATE" in s or "_helper" in s or "_internal" in s for s in surface)


def test_public_interface_broken_source_is_empty():
    assert ctx.public_interface("def broken(:") == []


# --- extraction against a real jj repo -----------------------------------------------


def _jj(cwd, *args):
    env = {**os.environ, "JJ_USER": "test", "JJ_EMAIL": "test@example.com"}
    res = subprocess.run(
        ["jj", *args], cwd=cwd, capture_output=True, text=True, timeout=30, env=env
    )
    assert res.returncode == 0, f"jj {' '.join(args)} failed: {res.stderr}"
    return res.stdout


@pytest.mark.skipif(shutil.which("jj") is None, reason="jj not installed")
def test_build_leaf_context_carries_landed_dep_interface(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _jj(repo, "git", "init")
    (repo / "temperature.py").write_text(
        "VALID_UNITS = frozenset(('C', 'F', 'K'))\n\n"
        "def convert(value: float, from_unit: str, to_unit: str) -> float:\n"
        "    return value\n"
    )
    (repo / "README.md").write_text("docs\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_temperature.py").write_text("def test_convert():\n    pass\n")
    _jj(repo, "describe", "-m", "auto: temperature module lands")
    commit = _jj(repo, "log", "--no-graph", "-r", "@", "-T", "commit_id.short()").strip()
    _jj(repo, "new")

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _journal(run_dir, _landed("Implement temperature module", commit))

    preamble = ctx.build_leaf_context(
        _task(deps=["Implement temperature module"]),
        run_dir=run_dir,
        repo=repo,
        epic_goal="Add temperature support",
        siblings=[
            _row("Implement temperature module", status="Done"),
            _row("Extend CLI router"),
            _row("Add integration test"),
        ],
    )

    assert "Add temperature support" in preamble
    # the landed dep's REAL interface, signatures and constants included
    assert "def convert(value: float, from_unit: str, to_unit: str) -> float" in preamble
    assert "VALID_UNITS" in preamble
    assert "README.md" in preamble  # non-Python files listed by path
    # test files are path-only: test functions are not contracts
    assert "tests/test_temperature.py" in preamble
    assert "def test_convert" not in preamble
    # scope fencing: other leaves titles-only, never the leaf itself or its deps
    assert "Add integration test" in preamble
    assert "do NOT implement" in preamble
    assert preamble.count("Extend CLI router") == 0


def test_build_leaf_context_no_deps_no_siblings_is_empty(tmp_path):
    preamble = ctx.build_leaf_context(
        _task(title="Solo leaf"),
        run_dir=tmp_path,
        repo=tmp_path,
        epic_goal="goal",
        siblings=[_row("Solo leaf")],
    )
    assert preamble == ""


def test_build_leaf_context_unlanded_dep_degrades_to_titles(tmp_path):
    """A dep with no landed commit contributes nothing — no crash, no stale guess."""
    preamble = ctx.build_leaf_context(
        _task(deps=["Never ran"]),
        run_dir=tmp_path,  # empty journal
        repo=tmp_path,  # not even a repo — commit lookup must not be reached
        epic_goal="goal",
        siblings=[_row("Other leaf")],
    )
    assert "Landed interfaces" not in preamble
    assert "Other leaf" in preamble


def test_build_leaf_context_truncates_whole_blocks_with_marker(tmp_path, monkeypatch):
    _journal(tmp_path, _landed("dep-a", "aaa"), _landed("dep-b", "bbb"))
    blocks = {"aaa": "x" * 3000, "bbb": "y" * 3000}
    monkeypatch.setattr(
        ctx, "_dep_block", lambda repo, title, commit: f'- "{title}":\n{blocks[commit]}'
    )
    preamble = ctx.build_leaf_context(
        _task(deps=["dep-a", "dep-b"]),
        run_dir=tmp_path,
        repo=tmp_path,
        epic_goal="goal",
        siblings=[],
        max_chars=4000,
    )
    assert len(preamble) < 4500
    assert "truncated: 1 more" in preamble  # dropped block is announced, never silent
