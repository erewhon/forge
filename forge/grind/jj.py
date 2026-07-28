"""jj operation-log checkpoints — commit-less time-travel for the grind loop.

jj auto-snapshots the working copy into its operation log on every command, so an "iteration
boundary" is just an op id, and rolling a bad iteration back is ``jj op restore <id>`` — no commits,
no bookmarks touched, nothing that lands. That is exactly what grind wants: keep the best working
state, discard regressions, and leave the repo's real history untouched.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_TIMEOUT = 30


class JJError(RuntimeError):
    """A jj operation failed, or the repo isn't a jj repo."""


def _run(args: list[str], repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["jj", "--no-pager", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
    )


def ensure_jj(repo: Path) -> Path:
    """Return the enclosing jj workspace root, or raise — grind's checkpoint/rollback needs the
    jj op log. Walks up from *repo* so grind works from a project subdirectory of a monorepo:
    the runbook's steps still run in *repo*, while jj snapshots/restores the whole workspace."""
    for candidate in (repo, *repo.parents):
        if (candidate / ".jj").is_dir():
            return candidate
    raise JJError(
        f"{repo} is not inside a jj repo. grind uses the jj operation log for commit-less "
        f"checkpoints; run it in (or under) a jj working copy (`jj git init` if needed)."
    )


def current_op(repo: Path) -> str:
    """The current operation id — a checkpoint to restore back to. Snapshots the working copy as a
    side effect (any jj command does), so calling this *after* an edit captures that edit."""
    result = _run(["op", "log", "--no-graph", "--limit", "1", "-T", "id.short()"], repo)
    if result.returncode != 0:
        raise JJError(f"jj op log failed: {result.stderr.strip()}")
    op = result.stdout.strip()
    if not op:
        raise JJError("jj op log returned no operation id")
    return op


def restore_op(repo: Path, op: str) -> None:
    """Restore the repo to operation *op* — reverts working-copy edits made since, no commit."""
    result = _run(["op", "restore", op], repo)
    if result.returncode != 0:
        raise JJError(f"jj op restore {op} failed: {result.stderr.strip()}")
