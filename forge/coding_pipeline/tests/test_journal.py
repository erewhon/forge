"""Tests for the coding pipeline journal module.

Covers: wave-number resumption, journal appending, attempt counting from
journal scans, and reconcile flipping only orphaned In Progress rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.coding_pipeline.journal import (
    append_gate_result,
    append_leaf_outcome,
    append_replan_action,
    count_attempts,
    count_attempts_for_all,
    failure_signature,
    load_wave,
    persist_wave,
    reconcile,
    recurring_failures,
    stuck_leaves,
)
from forge.coding_pipeline.models import LeafOutcome, SuiteResult, WaveRecord, WaveReport
from forge.task_worker.nous_client import nous_available

requires_nous = pytest.mark.skipif(
    not nous_available(),
    reason="exercises the Nous task-store path (install forge[nous])",
)

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
        from forge.coding_pipeline.journal import _highest_wave_number

        assert _highest_wave_number(epics_runs_dir, "nonexistent") == 0

    def test_next_wave_number_resumes_from_highest(self, epics_runs_dir: Path):
        from forge.coding_pipeline.journal import next_wave_number

        assert next_wave_number(epics_runs_dir, "my-epic") == 1  # fresh epic
        persist_wave(epics_runs_dir, "my-epic", WaveRecord(wave=3, report=WaveReport(wave=3)))
        assert next_wave_number(epics_runs_dir, "my-epic") == 4


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
    def test_flips_every_in_progress_task(self):
        """At orchestrator startup, ANY In Progress task for the feature is an orphan
        (single-orchestrator invariant) — the run dir persisting is irrelevant."""
        update_calls: list[tuple[str, str, str]] = []

        def fake_update(task: str, status: str, notes: str = "") -> None:
            update_calls.append((task, status, notes))

        result = reconcile(
            "toy-epic",
            in_progress=lambda prefix: ["Task: Add parser", "Task: Add writer"],
            update_status=fake_update,
        )
        assert result == ["Task: Add parser", "Task: Add writer"]
        assert [(c[0], c[1]) for c in update_calls] == [
            ("Task: Add parser", "Ready"),
            ("Task: Add writer", "Ready"),
        ]
        assert "crash recovery" in update_calls[0][2]

    def test_nothing_in_progress_is_a_noop(self):
        update_calls: list = []
        result = reconcile(
            "toy-epic",
            in_progress=lambda prefix: [],
            update_status=lambda *a, **k: update_calls.append(a),
        )
        assert result == []
        assert update_calls == []

    @requires_nous
    def test_epic_ref_prefix_passes_through_to_query(self):
        """reconcile scopes by the epic's external_ref prefix — the same membership
        rule as the wave planner, whatever Feature value a leaf carries."""
        seen: list[str] = []

        def fake_in_progress(prefix: str) -> list[str]:
            seen.append(prefix)
            return []

        reconcile("toy-epic", in_progress=fake_in_progress)
        assert seen == ["pipeline:toy-epic:"]


class TestLandedTitles:
    def test_reads_done_dispatches_only(self, tmp_path: Path):
        from forge.coding_pipeline.journal import append_leaf_outcome, landed_titles

        append_leaf_outcome(tmp_path, "won", LeafOutcome(leaf="won", status="done", commit_id="c"))
        append_leaf_outcome(tmp_path, "lost", LeafOutcome(leaf="lost", status="failed", reason="x"))
        append_leaf_outcome(tmp_path, "meh", LeafOutcome(leaf="meh", status="skipped"))
        assert landed_titles(tmp_path) == {"won"}

    def test_empty_without_journal(self, tmp_path: Path):
        from forge.coding_pipeline.journal import landed_titles

        assert landed_titles(tmp_path) == set()


# ---------------------------------------------------------------------------
# Failure signatures & the no-progress (Ralph-loop) guard
# ---------------------------------------------------------------------------


class TestFailureSignature:
    def test_same_reason_signs_identically(self):
        assert failure_signature("model reported BLOCKED: cannot import x") == failure_signature(
            "model reported BLOCKED: cannot import x"
        )

    def test_different_failure_modes_differ(self):
        assert failure_signature("tests red: test_parse failed") != failure_signature(
            "opencode failed: timeout"
        )

    def test_volatile_detail_is_normalized_away(self):
        # line numbers, durations, temp paths, and shas differ per run but the mode is the same
        a = "tests red at /tmp/abc123/repo/test_x.py:42 after 3.1s (deadbeef)"
        b = "tests red at /tmp/zzz999/repo/test_x.py:87 after 9.7s (cafef00d)"
        assert failure_signature(a) == failure_signature(b)

    def test_blank_reason_has_no_signature(self):
        assert failure_signature("") == ""
        assert failure_signature("   \n") == ""


class TestFailureSigJournalling:
    def test_failed_outcome_records_a_signature(self, run_dir: Path):
        append_leaf_outcome(run_dir, "x", LeafOutcome(leaf="x", status="failed", reason="boom"))
        rec = json.loads(run_dir.joinpath("journal.jsonl").read_text().strip())
        assert rec["failure_sig"] == failure_signature("boom")

    def test_done_outcome_has_no_signature(self, run_dir: Path):
        append_leaf_outcome(run_dir, "x", LeafOutcome(leaf="x", status="done", commit_id="c"))
        rec = json.loads(run_dir.joinpath("journal.jsonl").read_text().strip())
        assert "failure_sig" not in rec


class TestStuckLeaves:
    def _fail(self, run_dir: Path, leaf: str, reason: str) -> None:
        append_leaf_outcome(run_dir, leaf, LeafOutcome(leaf=leaf, status="failed", reason=reason))

    def test_two_identical_failures_are_stuck(self, run_dir: Path):
        self._fail(run_dir, "x", "assert 1 == 2")
        self._fail(run_dir, "x", "assert 1 == 2")
        assert stuck_leaves(run_dir, ["x"]) == {"x"}

    def test_two_different_failures_are_not_stuck(self, run_dir: Path):
        self._fail(run_dir, "x", "import error")
        self._fail(run_dir, "x", "assert 1 == 2")
        assert stuck_leaves(run_dir, ["x"]) == set()

    def test_single_failure_is_not_stuck(self, run_dir: Path):
        self._fail(run_dir, "x", "boom")
        assert stuck_leaves(run_dir, ["x"]) == set()

    def test_a_success_between_failures_resets_the_streak(self, run_dir: Path):
        self._fail(run_dir, "x", "boom")
        append_leaf_outcome(run_dir, "x", LeafOutcome(leaf="x", status="done", commit_id="c"))
        self._fail(run_dir, "x", "boom")
        assert stuck_leaves(run_dir, ["x"]) == set()

    def test_only_requested_leaves_are_considered(self, run_dir: Path):
        self._fail(run_dir, "x", "boom")
        self._fail(run_dir, "x", "boom")
        self._fail(run_dir, "y", "boom")
        self._fail(run_dir, "y", "boom")
        assert stuck_leaves(run_dir, ["x"]) == {"x"}

    def test_no_journal_is_not_stuck(self, tmp_path: Path):
        assert stuck_leaves(tmp_path, ["x"]) == set()


class TestRecurringFailures:
    def _fail(self, run_dir: Path, leaf: str, reason: str) -> None:
        append_leaf_outcome(run_dir, leaf, LeafOutcome(leaf=leaf, status="failed", reason=reason))

    def test_signature_seen_twice_recurs_across_leaves(self, run_dir: Path):
        # same failure MODE (only volatile detail — the line number — differs) on two leaves
        self._fail(run_dir, "a", "gate: ruff import-order on line 12")
        self._fail(run_dir, "b", "gate: ruff import-order on line 88")
        recurring = recurring_failures(run_dir)
        assert len(recurring) == 1
        sig, count, reason = recurring[0]
        assert count == 2
        assert sig == failure_signature("gate: ruff import-order on line 12")
        assert reason == "gate: ruff import-order on line 12"  # first-seen representative

    def test_single_occurrence_is_not_recurring(self, run_dir: Path):
        self._fail(run_dir, "a", "import error")
        self._fail(run_dir, "b", "different failure entirely")
        assert recurring_failures(run_dir) == []

    def test_done_outcomes_are_ignored(self, run_dir: Path):
        self._fail(run_dir, "a", "boom")
        append_leaf_outcome(run_dir, "a", LeafOutcome(leaf="a", status="done", commit_id="c"))
        assert recurring_failures(run_dir) == []

    def test_sorted_most_frequent_first(self, run_dir: Path):
        for _ in range(3):
            self._fail(run_dir, "x", "frequent failure")
        for _ in range(2):
            self._fail(run_dir, "y", "less frequent failure")
        counts = [c for _, c, _ in recurring_failures(run_dir)]
        assert counts == [3, 2]
