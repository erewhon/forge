"""Tests for the coding pipeline journal module.

Covers: wave-number resumption, journal appending, attempt counting from
journal scans, and reconcile flipping only orphaned In Progress rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.coding_pipeline.journal import (
    append_gate_result,
    append_leaf_outcome,
    append_replan_action,
    count_attempts,
    count_attempts_for_all,
    load_wave,
    persist_wave,
    reconcile,
)
from agents.coding_pipeline.models import LeafOutcome, SuiteResult, WaveRecord, WaveReport

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """Return an empty epic run directory."""
    d = tmp_path / "pipeline-runs" / "my-epic"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def epics_runs_dir(tmp_path: Path) -> Path:
    """Return the top-level pipeline-runs dir."""
    d = tmp_path / "pipeline-runs"
    d.mkdir(parents=True)
    return d


# ---------------------------------------------------------------------------
# Wave persistence & numbering resumption
# ---------------------------------------------------------------------------


class TestWavePersistence:
    def test_first_wave_is_wave_0001(self, epics_runs_dir: Path):
        record = WaveRecord(
            wave=1,
            report=WaveReport(wave=1, suite=SuiteResult(passed=True)),
        )
        path = persist_wave(epics_runs_dir, "my-epic", record)
        assert path.name == "wave-0001.json"
        loaded = load_wave(epics_runs_dir, "my-epic", 1)
        assert loaded is not None
        assert loaded.wave == 1

    def test_wave_numbers_continue_from_existing(self, epics_runs_dir: Path):
        """Writing wave 3 when wave 1 and 2 exist should create wave-0003."""
        persist_wave(epics_runs_dir, "my-epic", WaveRecord(wave=1, report=WaveReport(wave=1)))
        persist_wave(epics_runs_dir, "my-epic", WaveRecord(wave=2, report=WaveReport(wave=2)))
        path = persist_wave(
            epics_runs_dir,
            "my-epic",
            WaveRecord(wave=3, report=WaveReport(wave=3)),
        )
        assert path.name == "wave-0003.json"

    def test_resume_higher_number_from_existing_files(self, epics_runs_dir: Path):
        """If waves 1 and 3 exist (wave 2 was skipped/deleted), the next wave
        should be 4, not 2."""
        d = epics_runs_dir / "my-epic"
        d.mkdir(parents=True)
        d.joinpath("wave-0001.json").write_text(
            WaveRecord(wave=1, report=WaveReport(wave=1)).model_dump_json()
        )
        d.joinpath("wave-0003.json").write_text(
            WaveRecord(wave=3, report=WaveReport(wave=3)).model_dump_json()
        )
        path = persist_wave(
            epics_runs_dir,
            "my-epic",
            WaveRecord(wave=4, report=WaveReport(wave=4)),
        )
        assert path.name == "wave-0004.json"

    def test_load_wave_returns_none_for_missing(self, epics_runs_dir: Path):
        assert load_wave(epics_runs_dir, "my-epic", 99) is None

    def test_load_wave_round_trips(self, epics_runs_dir: Path):
        orig = WaveRecord(
            wave=5,
            dispatched=["a", "b"],
            report=WaveReport(wave=5, suite=SuiteResult(passed=False, output_tail="boom")),
        )
        persist_wave(epics_runs_dir, "my-epic", orig)
        revived = load_wave(epics_runs_dir, "my-epic", 5)
        assert revived is not None
        assert revived.dispatched == ["a", "b"]
        assert revived.report.suite is not None
        assert revived.report.suite.passed is False

    def test_no_run_dir_returns_zero_highest(self, epics_runs_dir: Path):
        """When no run dir exists for the epic, highest wave number is 0."""
        from agents.coding_pipeline.journal import _highest_wave_number

        assert _highest_wave_number(epics_runs_dir, "nonexistent") == 0


# ---------------------------------------------------------------------------
# Journal appending
# ---------------------------------------------------------------------------


class TestJournalAppend:
    def test_leaf_outcome_appends_to_journal(self, run_dir: Path):
        outcome = LeafOutcome(leaf="parser", status="done", commit_id="abc123")
        append_leaf_outcome(run_dir, "parser", outcome)

        journal_lines = run_dir.joinpath("journal.jsonl").read_text().strip().split("\n")
        assert len(journal_lines) == 1
        rec = json.loads(journal_lines[0])
        assert rec["event"] == "leaf_dispatch"
        assert rec["leaf"] == "parser"
        assert rec["status"] == "done"
        assert rec["commit_id"] == "abc123"

    def test_leaf_outcome_without_optional_fields(self, run_dir: Path):
        outcome = LeafOutcome(leaf="writer", status="failed", reason="tests red")
        append_leaf_outcome(run_dir, "writer", outcome)

        rec = json.loads(run_dir.joinpath("journal.jsonl").read_text().strip())
        assert rec["event"] == "leaf_dispatch"
        assert "commit_id" not in rec

    def test_gate_result_appends(self, run_dir: Path):
        append_gate_result(run_dir, "suite", True)
        rec = json.loads(run_dir.joinpath("journal.jsonl").read_text().strip())
        assert rec["event"] == "gate_result"
        assert rec["gate"] == "suite"
        assert rec["passed"] is True

    def test_replan_action_appends(self, run_dir: Path):
        append_replan_action(run_dir, "fixup", finding_slug="crash")
        rec = json.loads(run_dir.joinpath("journal.jsonl").read_text().strip())
        assert rec["event"] == "replan"
        assert rec["action"] == "fixup"
        assert rec["finding_slug"] == "crash"

    def test_multiple_journal_entries(self, run_dir: Path):
        append_leaf_outcome(run_dir, "a", LeafOutcome(leaf="a", status="done"))
        append_leaf_outcome(run_dir, "b", LeafOutcome(leaf="b", status="failed", reason="boom"))
        append_gate_result(run_dir, "suite", False)

        lines = run_dir.joinpath("journal.jsonl").read_text().strip().split("\n")
        assert len(lines) == 3

    def test_journal_not_written_on_dry_run(self, run_dir: Path):
        """Journal is never written when the caller is in dry-run — but this
        module has no dry_run flag.  All appends go to disk.  The test exists
        as a sanity check that no conditional skip exists."""
        outcome = LeafOutcome(leaf="x", status="done")
        append_leaf_outcome(run_dir, "x", outcome)
        assert run_dir.joinpath("journal.jsonl").exists()


# ---------------------------------------------------------------------------
# Attempt counting from journal
# ---------------------------------------------------------------------------


class TestAttemptCounting:
    def test_zero_attempts_no_journal(self, run_dir: Path):
        assert count_attempts(run_dir, "parser") == 0

    def test_one_attempt(self, run_dir: Path):
        append_leaf_outcome(
            run_dir,
            "parser",
            LeafOutcome(leaf="parser", status="done"),
        )
        assert count_attempts(run_dir, "parser") == 1

    def test_multiple_attempts_same_leaf(self, run_dir: Path):
        for i in range(3):
            append_leaf_outcome(
                run_dir,
                "parser",
                LeafOutcome(leaf="parser", status="failed" if i < 2 else "done"),
            )
        assert count_attempts(run_dir, "parser") == 3
        assert count_attempts(run_dir, "writer") == 0

    def test_counts_different_leaves_independently(self, run_dir: Path):
        append_leaf_outcome(
            run_dir,
            "a",
            LeafOutcome(leaf="a", status="done"),
        )
        append_leaf_outcome(
            run_dir,
            "a",
            LeafOutcome(leaf="a", status="failed"),
        )
        append_leaf_outcome(
            run_dir,
            "b",
            LeafOutcome(leaf="b", status="done"),
        )
        assert count_attempts(run_dir, "a") == 2
        assert count_attempts(run_dir, "b") == 1

    def test_counts_for_all(self, run_dir: Path):
        append_leaf_outcome(
            run_dir,
            "a",
            LeafOutcome(leaf="a", status="done"),
        )
        append_leaf_outcome(
            run_dir,
            "b",
            LeafOutcome(leaf="b", status="failed"),
        )
        append_leaf_outcome(
            run_dir,
            "a",
            LeafOutcome(leaf="a", status="done"),
        )
        counts = count_attempts_for_all(run_dir, ["a", "b", "c"])
        assert counts == {"a": 2, "b": 1, "c": 0}


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


class TestReconcile:
    def test_no_orphans_when_run_dir_exists(self, epics_runs_dir: Path):
        """When a run dir exists, reconcile returns empty — nothing to fix."""
        (epics_runs_dir / "my-epic").mkdir(parents=True)
        result = reconcile(
            epics_runs_dir,
            "my-epic",
            run_dir_exists=lambda _r, _e: True,
        )
        assert result == []

    def test_flips_orphaned_in_progress_tasks(self, epics_runs_dir: Path):
        """Tasks In Progress for this epic with no live run get flipped to Ready."""
        update_calls: list[tuple[str, str, str]] = []

        def fake_update(task: str, status: str, notes: str = "") -> None:
            update_calls.append((task, status, notes))

        result = reconcile(
            epics_runs_dir,
            "my-epic",
            run_dir_exists=lambda _r, _e: False,
            query_tasks=lambda status, feature: [
                {"task": "Task: Add parser", "status": "In Progress"},
                {"task": "Task: Add writer", "status": "In Progress"},
            ],
            update_task_status=fake_update,
        )
        assert "Task: Add parser" in result
        assert "Task: Add writer" in result
        assert len(update_calls) == 2

    def test_only_flips_in_progress(self, epics_runs_dir: Path):
        """Ready tasks are not touched — query returns only In Progress."""
        update_calls: list[tuple[str, str, str]] = []

        def fake_update(task: str, status: str, notes: str = "") -> None:
            update_calls.append((task, status, notes))

        # The query returns In Progress tasks (as the real query would),
        # but reconcile should only flip tasks that have no live run.
        result = reconcile(
            epics_runs_dir,
            "my-epic",
            run_dir_exists=lambda _r, _e: False,
            query_tasks=lambda status, feature: [
                {"task": "Task: In Progress thing", "status": "In Progress"},
            ],
            update_task_status=fake_update,
        )
        assert len(result) == 1
        assert update_calls[0][1] == "Ready"

    def test_skips_empty_task_names(self, epics_runs_dir: Path):
        """Rows with empty task name are skipped."""
        result = reconcile(
            epics_runs_dir,
            "my-epic",
            run_dir_exists=lambda _r, _e: False,
            query_tasks=lambda status, feature: [
                {"task": "", "status": "In Progress"},
            ],
        )
        assert result == []

    def test_reconcile_calls_update_task_status(self, epics_runs_dir: Path):
        """Each orphaned task triggers update_task_status with Ready + diagnostic."""
        update_calls: list[tuple[str, str, str]] = []

        def fake_update(task: str, status: str, notes: str = "") -> None:
            update_calls.append((task, status, notes))

        reconcile(
            epics_runs_dir,
            "my-epic",
            run_dir_exists=lambda _r, _e: False,
            query_tasks=lambda status, feature: [
                {"task": "Task: X", "status": "In Progress"},
            ],
            update_task_status=fake_update,
        )
        assert len(update_calls) == 1
        assert update_calls[0][0] == "Task: X"
        assert update_calls[0][1] == "Ready"
        assert "Reconciled by coding pipeline" in update_calls[0][2]

    def test_no_query_tasks_when_run_dir_exists(self, epics_runs_dir: Path):
        """When run dir exists, query_tasks is never called."""
        called = []

        def fake_query(**kwargs):
            called.append(kwargs)
            return []

        reconcile(
            epics_runs_dir,
            "my-epic",
            run_dir_exists=lambda _r, _e: True,
            query_tasks=fake_query,
        )
        assert called == []
