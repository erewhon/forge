"""jj workspace lifecycle for parallel edit candidates.

Each candidate model runs in its own jj workspace sharing the same underlying repo.
Workspaces are created at a pinned base revision so diffs are stable regardless of
later activity in the user's main workspace.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import uuid
from pathlib import Path

from agents.parallel_edit.config import settings
from agents.parallel_edit.models import DiffStat

_STAT_SUMMARY_RE = re.compile(
    r"(\d+)\s+files?\s+changed"
    r"(?:,\s+(\d+)\s+insertions?\(\+\))?"
    r"(?:,\s+(\d+)\s+deletions?\(-\))?"
)


class JJError(RuntimeError):
    """Raised when a jj command fails."""


def _run_jj(args: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["jj", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise JJError(
            f"jj {' '.join(args)} failed (exit {result.returncode}):\n{result.stderr.strip()}"
        )
    return result


def resolve_base_rev(repo: Path, rev: str = "@") -> str:
    """Resolve a jj revset to a stable commit_id, so later diffs anchor to the same point."""
    result = _run_jj(
        ["log", "-r", rev, "--no-graph", "-T", "commit_id", "--limit", "1"],
        cwd=repo,
    )
    commit_id = result.stdout.strip()
    if not commit_id:
        raise JJError(f"could not resolve revset {rev!r} in {repo}")
    return commit_id


def workspace_destination(repo: Path, label: str) -> Path:
    parent = settings.workspace_base_dir or repo.parent
    name = f"{settings.workspace_name_prefix}-{label}-{uuid.uuid4().hex[:6]}"
    return parent / name


def create_workspace(repo: Path, dest: Path, *, base_rev: str) -> None:
    """Create a new jj workspace at `dest` with @ on top of `base_rev`."""
    if dest.exists():
        raise JJError(f"workspace destination already exists: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run_jj(
        ["workspace", "add", "--revision", base_rev, str(dest)],
        cwd=repo,
    )


def collect_diff(workspace: Path, base_rev: str) -> tuple[str, DiffStat]:
    """Collect the unified diff and a parsed DiffStat for everything since base_rev."""
    diff_proc = _run_jj(
        ["diff", "--from", base_rev, "--git"],
        cwd=workspace,
    )
    stat_proc = _run_jj(
        ["diff", "--from", base_rev, "--stat"],
        cwd=workspace,
    )
    return diff_proc.stdout, _parse_diff_stat(stat_proc.stdout)


def _parse_diff_stat(stat_output: str) -> DiffStat:
    """Parse the trailing 'N files changed, X insertions(+), Y deletions(-)' line."""
    if not stat_output.strip():
        return DiffStat()
    # The summary is always the last non-empty line of --stat output
    last_line = ""
    for line in reversed(stat_output.splitlines()):
        if line.strip():
            last_line = line.strip()
            break
    match = _STAT_SUMMARY_RE.search(last_line)
    if not match:
        return DiffStat()
    files = int(match.group(1))
    insertions = int(match.group(2) or 0)
    deletions = int(match.group(3) or 0)
    return DiffStat(files_changed=files, insertions=insertions, deletions=deletions)


def forget_workspace(repo: Path, workspace: Path) -> None:
    """Tell jj to forget the workspace, then remove the directory."""
    name = workspace.name
    # check=False because we still want to rmtree even if forget fails
    _run_jj(["workspace", "forget", name], cwd=repo, check=False)
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
