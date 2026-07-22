"""Home-repo delta detection and the switch-back briefing render."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from forge.shared.baton import Baton
from forge.switcheroo.models import FailoverLog, LeafOutcome
from forge.switcheroo.reconcile import (
    HomeChanges,
    HomeCommit,
    home_repo_changes,
    render_switchback,
)


def _has_jj() -> bool:
    try:
        return subprocess.run(["jj", "--version"], capture_output=True).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


# --- home_repo_changes -----------------------------------------------------


def test_no_anchor_is_graceful(tmp_path: Path):
    changes = home_repo_changes(tmp_path, None)
    assert changes.commits == [] and changes.changed_files == []
    assert "no VCS anchor" in changes.note


def test_non_repo_is_graceful(tmp_path: Path):
    changes = home_repo_changes(tmp_path, "deadbeef")
    assert changes.note  # some explanation, no crash


@pytest.mark.skipif(not _has_jj(), reason="jj not installed")
def test_jj_reports_commits_and_files_since_anchor(tmp_path: Path):
    def jj(*args: str) -> None:
        subprocess.run(["jj", "--no-pager", *args], cwd=tmp_path, capture_output=True, check=True)

    jj("git", "init")
    (tmp_path / "a.txt").write_text("hello")
    jj("describe", "-m", "pre-failover work")
    anchor = subprocess.run(
        ["jj", "--no-pager", "log", "-r", "@", "--no-graph", "-T", "change_id.short()"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    ).stdout.strip()
    jj("new")
    (tmp_path / "b.txt").write_text("fleet work")
    jj("describe", "-m", "fleet leaf: add b")
    jj("new")

    (tmp_path / ".forge").mkdir()
    (tmp_path / ".forge" / "baton.md").write_text("machinery churn")  # must be filtered out

    changes = home_repo_changes(tmp_path, anchor)
    assert changes.vcs == "jj"
    assert "b.txt" in changes.changed_files
    assert not any(f.startswith(".forge/") for f in changes.changed_files)  # own churn filtered
    # The described fleet commit shows; the empty tip working-copy commit is filtered out.
    assert any("fleet leaf: add b" in c.description for c in changes.commits)
    assert all(c.description.strip() for c in changes.commits)


# --- render_switchback -----------------------------------------------------


def test_render_combines_all_three_sources():
    baton = Baton(
        goal="ship switcheroo",
        next_action="build switch-back",
        plan=["reconcile", "continue"],
        decisions=["baton is a shared primitive"],
    )
    log = FailoverLog(
        started_at="2026-07-21T00:00:00+00:00",
        ended_at="2026-07-21T01:00:00+00:00",
        model_tier="auto-free",
        baton_change_id="anch01",
        outcomes=[LeafOutcome(task="t1", project="P", status="done", commit_id="c1")],
    )
    changes = HomeChanges(
        vcs="jj",
        anchor="anch01",
        commits=[HomeCommit(change_id="x1", description="home leaf")],
        changed_files=["main.py"],
    )
    out = render_switchback(baton, log, changes)
    assert "ship switcheroo" in out  # where we were
    assert "P / t1" in out and "c1" in out  # what the fleet did
    assert "home leaf" in out and "main.py" in out  # home-repo delta
    assert "Resume:" in out  # how to continue


def test_render_notes_when_no_home_changes():
    out = render_switchback(Baton(goal="g"), FailoverLog(started_at="t"), HomeChanges(anchor="a"))
    assert "no home-repo changes" in out


def test_render_survives_missing_baton_and_log():
    out = render_switchback(None, None, HomeChanges(note="not a repo"))
    assert "no failover window on record" in out
    assert "not a repo" in out
