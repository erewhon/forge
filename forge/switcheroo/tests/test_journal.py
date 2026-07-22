"""Failover journal — incremental persistence, prior-window archival, and the switchback summary."""

from __future__ import annotations

from pathlib import Path

from forge.shared.baton import Baton
from forge.switcheroo.journal import (
    end_failover,
    failover_path,
    history_dir,
    read_failover,
    record_outcome,
    render_failover_summary,
    start_failover,
)
from forge.switcheroo.models import FailoverLog, LeafOutcome


def _baton() -> Baton:
    return Baton(goal="ship switcheroo", vcs="jj", change_id="abc123")


def test_read_absent_is_none(tmp_path: Path):
    assert read_failover(tmp_path) is None


def test_start_creates_window_anchored_to_baton(tmp_path: Path):
    log = start_failover(tmp_path, baton=_baton(), model_tier="auto-free", reason="all agents 529")
    assert failover_path(tmp_path).is_file()
    assert log.baton_goal == "ship switcheroo"
    assert log.baton_change_id == "abc123"
    assert log.model_tier == "auto-free"
    assert log.reason == "all agents 529"
    assert log.ended_at is None


def test_record_outcome_persists_incrementally(tmp_path: Path):
    start_failover(tmp_path, baton=_baton(), model_tier="auto")
    record_outcome(tmp_path, LeafOutcome(task="t1", project="P", status="done", commit_id="c1"))
    # A *fresh read* sees it — durability across an interruption is the whole point.
    reread = read_failover(tmp_path)
    assert [o.task for o in reread.outcomes] == ["t1"]
    assert reread.done[0].commit_id == "c1"


def test_end_stamps_ended_at(tmp_path: Path):
    start_failover(tmp_path, baton=_baton(), model_tier="auto")
    closed = end_failover(tmp_path)
    assert closed is not None and closed.ended_at is not None


def test_end_with_no_window_is_none(tmp_path: Path):
    assert end_failover(tmp_path) is None


def test_starting_new_window_archives_the_prior_one(tmp_path: Path):
    first = start_failover(tmp_path, baton=_baton(), model_tier="auto")
    record_outcome(tmp_path, LeafOutcome(task="t1", project="P", status="done"))
    start_failover(tmp_path, baton=_baton(), model_tier="auto")  # second window

    archived = list(history_dir(tmp_path).glob("*.json"))
    assert len(archived) == 1
    prior = FailoverLog.model_validate_json(archived[0].read_text())
    assert prior.started_at == first.started_at
    assert [o.task for o in prior.outcomes] == ["t1"]
    # The active window is the fresh one.
    assert read_failover(tmp_path).outcomes == []


def test_record_tolerates_missing_window(tmp_path: Path):
    # No start_failover() first: a crash-then-record path must not lose the result.
    record_outcome(tmp_path, LeafOutcome(task="t1", project="P", status="failed", reason="boom"))
    assert read_failover(tmp_path).failed[0].reason == "boom"


def test_summary_reports_each_bucket_and_switchback_anchor(tmp_path: Path):
    log = FailoverLog(
        started_at="2026-07-21T00:00:00+00:00",
        ended_at="2026-07-21T01:00:00+00:00",
        model_tier="auto-free",
        baton_goal="ship switcheroo",
        baton_change_id="abc123",
        outcomes=[
            LeafOutcome(task="t1", project="P", status="done", commit_id="c1", changed_files=["a"]),
            LeafOutcome(task="t2", project="Q", status="failed", reason="tests failed"),
            LeafOutcome(task="t3", project="P", status="skipped", reason="gate"),
        ],
    )
    out = render_failover_summary(log)
    assert "1 done · 1 failed · 1 skipped" in out
    assert "P / t1" in out and "c1" in out
    assert "tests failed" in out
    assert "jj diff" in out and "abc123" in out
