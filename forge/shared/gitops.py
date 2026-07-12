"""Thin git plumbing shared by the fleet agents — explicit refs and paths, no cwd magic.

The agent NEVER touches the caller's working copy: merge work happens in a temporary
``git worktree`` (created here, always removed), so a dirty checkout — or a jj-colocated
repo whose working copy jj owns — is never at risk. jj colocated repos are plain git repos
underneath; new refs created here are picked up by the next jj command.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

_TIMEOUT = 120


class GitError(RuntimeError):
    """A git invocation failed; the message carries the command and stderr."""


def git(repo: Path, *args: str, timeout: int = _TIMEOUT, check: bool = True) -> str:
    """Run ``git <args>`` in *repo*; return stripped stdout. Non-zero raises GitError."""
    result = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout.strip()


def git_ok(repo: Path, *args: str) -> bool:
    """True if ``git <args>`` exits 0 — for existence probes."""
    try:
        git(repo, *args)
        return True
    except GitError:
        return False


def detect_branch(repo: Path, ref_namespace: str) -> str:
    """First of main/master that exists under *ref_namespace* (e.g. ``refs/heads``,
    ``refs/remotes/upstream``). Raises GitError naming the settings to set instead."""
    for candidate in ("main", "master"):
        if git_ok(repo, "rev-parse", "--verify", "--quiet", f"{ref_namespace}/{candidate}"):
            return candidate
    raise GitError(
        f"no main/master under {ref_namespace} — set UPSTREAM_SYNC_UPSTREAM_BRANCH / "
        "UPSTREAM_SYNC_LOCAL_BRANCH explicitly"
    )


@contextmanager
def temporary_worktree(repo: Path, branch: str, start_point: str):
    """A disposable worktree with *branch* (re)created at *start_point*.

    ``-B`` deliberately resets an existing local sync branch: re-running the sync
    regenerates the branch rather than erroring on yesterday's leftover. The worktree is
    force-removed on exit no matter what happened inside.
    """
    parent = Path(tempfile.mkdtemp(prefix="upstream-sync-"))
    worktree = parent / "wt"
    git(repo, "worktree", "add", "-B", branch, str(worktree), start_point)
    try:
        yield worktree
    finally:
        git(repo, "worktree", "remove", "--force", str(worktree), check=False)
        shutil.rmtree(parent, ignore_errors=True)
