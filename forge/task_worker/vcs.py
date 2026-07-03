"""VCS detection, diff inspection, commit, and revert for jj and git repos."""

from __future__ import annotations

import subprocess
from pathlib import Path

_TIMEOUT = 30
# Safety cap: never `git clean` more than this many untracked files.
_CLEAN_SAFETY_LIMIT = 20


class VCSError(RuntimeError):
    """A VCS operation failed."""


def _run(cmd: list[str], cwd: Path, timeout: int = _TIMEOUT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )


def detect_vcs(repo_path: Path) -> str:
    """Return 'jj', 'git', or '' based on what's in the repo.

    Prefers jj if both are present (same policy as the code_reviewer collector).
    """
    if (repo_path / ".jj").is_dir():
        return "jj"
    if (repo_path / ".git").exists():
        return "git"
    return ""


# ---------------------------------------------------------------------------
# Changed files
# ---------------------------------------------------------------------------


def _git_changed_files(repo_path: Path) -> list[str]:
    """All files with any kind of change in the working copy (staged + unstaged + untracked)."""
    # Tracked (staged or unstaged) vs HEAD
    tracked = _run(["git", "diff", "--name-only", "HEAD"], repo_path)
    if tracked.returncode != 0:
        raise VCSError(f"git diff failed: {tracked.stderr.strip()}")

    # Untracked (not ignored)
    untracked = _run(["git", "ls-files", "--others", "--exclude-standard"], repo_path)
    if untracked.returncode != 0:
        raise VCSError(f"git ls-files failed: {untracked.stderr.strip()}")

    names: list[str] = []
    for block in (tracked.stdout, untracked.stdout):
        for line in block.splitlines():
            line = line.strip()
            if line and line not in names:
                names.append(line)
    return names


def _jj_changed_files(repo_path: Path) -> list[str]:
    """Files changed in the current jj working copy vs its parent."""
    result = _run(
        ["jj", "diff", "--no-pager", "--name-only"],
        repo_path,
    )
    if result.returncode != 0:
        raise VCSError(f"jj diff failed: {result.stderr.strip()}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def get_changed_files(repo_path: Path) -> list[str]:
    """Return the list of files changed in the working copy.

    For jj repos: uses the current change's diff vs parent.
    For git repos: includes staged + unstaged + untracked files.
    """
    vcs = detect_vcs(repo_path)
    if vcs == "jj":
        return _jj_changed_files(repo_path)
    if vcs == "git":
        return _git_changed_files(repo_path)
    raise VCSError(f"No VCS detected in {repo_path}")


# ---------------------------------------------------------------------------
# Revert
# ---------------------------------------------------------------------------


def _jj_revert(repo_path: Path) -> None:
    """Restore the working copy to match the parent commit."""
    result = _run(["jj", "restore"], repo_path)
    if result.returncode != 0:
        raise VCSError(f"jj restore failed: {result.stderr.strip()}")


def _git_revert(repo_path: Path) -> None:
    """Reset tracked changes and remove untracked files — with a safety limit."""
    # Discard tracked changes (staged + unstaged)
    checkout = _run(["git", "checkout", "--", "."], repo_path)
    if checkout.returncode != 0:
        raise VCSError(f"git checkout failed: {checkout.stderr.strip()}")
    # Unstage anything still staged
    reset = _run(["git", "reset", "HEAD", "--", "."], repo_path)
    if reset.returncode != 0:
        # Not fatal — there may have been nothing staged.
        pass

    # Count untracked before cleaning
    untracked = _run(["git", "ls-files", "--others", "--exclude-standard"], repo_path)
    if untracked.returncode != 0:
        raise VCSError(f"git ls-files failed: {untracked.stderr.strip()}")
    untracked_files = [line.strip() for line in untracked.stdout.splitlines() if line.strip()]
    if len(untracked_files) > _CLEAN_SAFETY_LIMIT:
        raise VCSError(
            f"Refusing to clean {len(untracked_files)} untracked files "
            f"(limit {_CLEAN_SAFETY_LIMIT}). Investigate manually."
        )
    if untracked_files:
        clean = _run(["git", "clean", "-fd"], repo_path)
        if clean.returncode != 0:
            raise VCSError(f"git clean failed: {clean.stderr.strip()}")


def revert_changes(repo_path: Path) -> None:
    """Revert working copy changes. Caller should call this before marking Ready."""
    vcs = detect_vcs(repo_path)
    if vcs == "jj":
        _jj_revert(repo_path)
    elif vcs == "git":
        _git_revert(repo_path)
    else:
        raise VCSError(f"No VCS detected in {repo_path}")


# ---------------------------------------------------------------------------
# Commit
# ---------------------------------------------------------------------------


def _jj_commit(repo_path: Path, message: str) -> str:
    """Describe the current change, then advance with `jj new`.

    Returns the short change id of the described commit.
    """
    # Describe the current working-copy change
    describe = _run(["jj", "describe", "-m", message], repo_path)
    if describe.returncode != 0:
        raise VCSError(f"jj describe failed: {describe.stderr.strip()}")

    # Capture the change id of the now-described commit (it is still @)
    id_result = _run(
        ["jj", "log", "--no-pager", "-r", "@", "--no-graph", "-T", "change_id.short()"],
        repo_path,
    )
    change_id = id_result.stdout.strip() if id_result.returncode == 0 else ""

    # Advance to a new empty working-copy change
    new_result = _run(["jj", "new"], repo_path)
    if new_result.returncode != 0:
        raise VCSError(f"jj new failed: {new_result.stderr.strip()}")

    return change_id


def _git_commit(repo_path: Path, message: str) -> str:
    """Stage everything and create a commit. Returns the short sha."""
    add = _run(["git", "add", "-A"], repo_path)
    if add.returncode != 0:
        raise VCSError(f"git add failed: {add.stderr.strip()}")

    commit = _run(["git", "commit", "-m", message], repo_path)
    if commit.returncode != 0:
        raise VCSError(f"git commit failed: {commit.stderr.strip()}")

    sha = _run(["git", "rev-parse", "--short", "HEAD"], repo_path)
    return sha.stdout.strip() if sha.returncode == 0 else ""


def commit(repo_path: Path, message: str) -> str:
    """Commit all working-copy changes. Returns a short commit/change id."""
    vcs = detect_vcs(repo_path)
    if vcs == "jj":
        return _jj_commit(repo_path, message)
    if vcs == "git":
        return _git_commit(repo_path, message)
    raise VCSError(f"No VCS detected in {repo_path}")
