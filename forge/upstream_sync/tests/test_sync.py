"""The sync loop against real git repos — every path: up-to-date, dry-run, clean merge,
conflict, gate miss, collision block, and the three --auto-merge outcomes."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from forge.upstream_sync import sync as sy
from forge.upstream_sync.config import settings
from forge.upstream_sync.models import CollisionFinding, CollisionVerdict
from forge.upstream_sync.tests.conftest import g, upstream_commit


def _clear() -> CollisionVerdict:
    return CollisionVerdict(collision=False)


@pytest.fixture
def loop(monkeypatch, tmp_path):
    """Happy-path collaborators: green suite, clear seat, captured advisory emission."""
    mocks = {
        "run_tests": MagicMock(return_value=(True, "ok")),
        "collision_verdict": MagicMock(return_value=_clear()),
        "emit_advisory": MagicMock(return_value=None),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(sy, name, mock)
    monkeypatch.setattr(settings, "auto_log_path", tmp_path / "upstream.jsonl")
    return mocks


def test_up_to_date_when_no_new_upstream_commits(repos, loop):
    _, fork, _ = repos
    result = sy.sync_upstream(fork, log=lambda m: None)
    assert result.status == "up-to-date"
    loop["run_tests"].assert_not_called()


def test_dry_run_compares_without_writes(repos, loop):
    upstream, fork, _ = repos
    upstream_commit(upstream, "feature.txt", "new\n", "upstream: add feature")
    result = sy.sync_upstream(fork, dry_run=True, log=lambda m: None)
    assert result.status == "planned"
    assert result.commits_behind == 1
    assert result.branch.startswith("upstream-sync/")
    assert "SPRINKLES.md" in result.layer.added
    assert "README.md" in result.layer.modified
    assert result.overlap == []  # feature.txt touches nothing the fork owns
    # No writes: the sync branch does not exist, no worktree lingers.
    assert g(fork, "branch", "--list", result.branch) == ""
    assert len(g(fork, "worktree", "list").splitlines()) == 1


def test_clean_merge_pushes_branch_and_leaves_working_copy_alone(repos, loop):
    upstream, fork, bare = repos
    upstream_commit(upstream, "feature.txt", "new\n", "upstream: add feature")
    # A dirty working copy must survive untouched — all work happens in a worktree.
    (fork / "wip.txt").write_text("uncommitted work\n")
    head_before = g(fork, "rev-parse", "HEAD")

    result = sy.sync_upstream(fork, log=lambda m: None)

    assert result.status == "branched"
    assert result.tests_passed is True
    branch_sha = g(bare, "rev-parse", f"refs/heads/{result.branch}")
    parents = g(fork, "rev-list", "--parents", "-n", "1", branch_sha).split()
    assert len(parents) == 3  # merge commit: itself + two parents
    assert (fork / "wip.txt").read_text() == "uncommitted work\n"
    assert g(fork, "rev-parse", "HEAD") == head_before
    assert len(g(fork, "worktree", "list").splitlines()) == 1  # cleaned up
    loop["emit_advisory"].assert_not_called()  # green branch push is a report, not a task


def test_conflict_files_task_and_pushes_nothing(repos, loop):
    upstream, fork, bare = repos
    upstream_commit(
        upstream, "README.md", "# soft serve\nupstream rewrote this\n", "upstream: rewrite README"
    )
    result = sy.sync_upstream(fork, project="P", log=lambda m: None)

    assert result.status == "conflict"
    assert result.conflicted == ["README.md"]
    assert result.branch is None
    assert "refs/heads/upstream-sync" not in g(bare, "for-each-ref")
    loop["emit_advisory"].assert_called_once()
    assert len(g(fork, "worktree", "list").splitlines()) == 1
    assert g(fork, "status", "--porcelain") == ""  # fork left pristine


def test_red_suite_still_pushes_branch_but_files_task(repos, loop):
    upstream, fork, bare = repos
    upstream_commit(upstream, "feature.txt", "new\n", "upstream: add feature")
    loop["run_tests"].return_value = (False, "1 failed")

    result = sy.sync_upstream(fork, project="P", log=lambda m: None)

    assert result.status == "advisory"
    assert "green-suite gate failed" in result.reason
    assert g(bare, "rev-parse", f"refs/heads/{result.branch}")  # reviewable branch exists
    loop["emit_advisory"].assert_called_once()


def test_collision_verdict_blocks_with_cited_finding(repos, loop):
    upstream, fork, _ = repos
    upstream_commit(upstream, "core.txt", "line1\nrefactored\nline3\n", "upstream: refactor")
    loop["collision_verdict"].return_value = CollisionVerdict(
        collision=True,
        findings=[CollisionFinding(file="core.txt", reason="layer wraps this")],
    )
    result = sy.sync_upstream(fork, project="P", log=lambda m: None)
    assert result.status == "advisory"
    assert "collision seat blocked" in result.reason
    assert "core.txt" in result.reason


def test_auto_merge_advances_origin_default_branch(repos, loop):
    upstream, fork, bare = repos
    upstream_commit(upstream, "feature.txt", "new\n", "upstream: add feature")
    local_main_before = g(fork, "rev-parse", "main")

    result = sy.sync_upstream(fork, auto_merge=True, log=lambda m: None)

    assert result.status == "merged" and result.merged_to_main
    assert g(bare, "rev-parse", "main") == g(bare, "rev-parse", f"refs/heads/{result.branch}")
    # The local default branch is deliberately NOT moved — the caller pulls.
    assert g(fork, "rev-parse", "main") == local_main_before
    assert "behind" in result.reason


def test_auto_merge_blocked_by_unknown_collision_verdict(repos, loop):
    upstream, fork, bare = repos
    upstream_commit(upstream, "feature.txt", "new\n", "upstream: add feature")
    loop["collision_verdict"].return_value = CollisionVerdict(
        collision=None, notes="seat unavailable"
    )
    result = sy.sync_upstream(fork, auto_merge=True, log=lambda m: None)
    assert result.status == "branched" and not result.merged_to_main
    assert "collision verdict unknown" in result.reason
    assert g(bare, "rev-parse", "main") != g(bare, "rev-parse", f"refs/heads/{result.branch}")


def test_auto_merge_blocked_when_local_and_origin_diverge(repos, loop):
    upstream, fork, _ = repos
    upstream_commit(upstream, "feature.txt", "new\n", "upstream: add feature")
    # Local main gains an unpushed commit — origin/main no longer matches.
    (fork / "local-only.txt").write_text("x\n")
    g(fork, "add", "-A")
    g(fork, "commit", "-qm", "local-only change")

    result = sy.sync_upstream(fork, auto_merge=True, log=lambda m: None)
    assert result.status == "branched" and not result.merged_to_main
    assert "reconcile" in result.reason


def test_rerun_against_same_upstream_state_is_idempotent(repos, loop):
    upstream, fork, _ = repos
    upstream_commit(upstream, "feature.txt", "new\n", "upstream: add feature")
    first = sy.sync_upstream(fork, log=lambda m: None)
    second = sy.sync_upstream(fork, log=lambda m: None)
    assert first.status == second.status == "branched"
    assert first.branch == second.branch  # -B + force push regenerate, not stack


def test_missing_upstream_remote_is_a_clear_error(repos, loop):
    _, fork, _ = repos
    g(fork, "remote", "remove", "upstream")
    result = sy.sync_upstream(fork, log=lambda m: None)
    assert result.status == "error"
    assert "git remote add upstream" in result.reason


def test_decision_log_records_every_run(repos, loop, tmp_path):
    _, fork, _ = repos
    sy.sync_upstream(fork, log=lambda m: None)
    assert (tmp_path / "upstream.jsonl").exists()
