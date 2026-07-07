"""jj workspace lifecycle shared by parallel_edit and the concurrent coding pipeline.

Each candidate (or leaf spec) runs in its own jj workspace sharing the same underlying repo.
Workspaces are created at a pinned base revision so diffs are stable regardless of later
activity in the user's main workspace.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import uuid
from pathlib import Path

from pydantic import BaseModel

__all__ = [
    "JJError",
    "_run_jj",
    "resolve_base_rev",
    "create_workspace",
    "ensure_git_marker",
    "collect_diff",
    "forget_workspace",
    "workspace_destination",
]


# --------------------------------------------------------------------------- data models


class DiffStat(BaseModel):
    files_changed: int = 0
    insertions: int = 0
    deletions: int = 0


# --------------------------------------------------------------------------- constants

_STAT_SUMMARY_RE = re.compile(
    r"(\d+)\s+files?\s+changed"
    r"(?:,\s+(\d+)\s+insertions?\(\+\))?"
    r"(?:,\s+(\d+)\s+deletions?\(-\))?"
)

# jj fileset terms excluded from every collected diff: tool-/run-generated cruft that isn't part
# of the candidate's actual change, so the judge scores the real edit and not the noise. Each term
# is negated and ANDed in `_diff_exclude_fileset()`.
#   - ".open-mem": opencode's open-mem plugin writes a local cache dir at the working-dir root on
#     every run (belt-and-braces alongside the global gitignore; `--pure` also disables the plugin).
#   - __pycache__ / *.pyc: Python bytecode a candidate generates by running or importing its own
#     code. In a repo without a matching `.gitignore` it leaks into the diff and gets mis-scored as
#     scope sprawl (observed live: a candidate penalized for a committed `.pyc`). The globs match at
#     any depth.
_DIFF_EXCLUDE_FILESETS = (
    '".open-mem"',
    'glob:"**/__pycache__/**"',
    'glob:"**/*.pyc"',
)


# --------------------------------------------------------------------------- subprocess helpers


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


# --------------------------------------------------------------------------- public API


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


def workspace_destination(
    repo: Path,
    label: str,
    *,
    base_dir: Path | None = None,
    prefix: str = "ws",
) -> Path:
    """Return the filesystem path where a workspace should be created.

    Parameters
    ----------
    repo:
        Path to the shared jj repo.
    label:
        Human-readable label for the workspace.
    base_dir:
        Parent directory for workspace directories. Defaults to *repo*'s parent.
    prefix:
        Prefix for the workspace name. The name format is ``{prefix}-{label}-{uuid}``.
    """
    parent = base_dir or repo.parent
    name = f"{prefix}-{label}-{uuid.uuid4().hex[:6]}"
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


def ensure_git_marker(workspace: Path) -> None:
    """Give the workspace a `.git` so tools that detect the project via git (opencode) work.

    A jj workspace contains only `.jj`; opencode walks up looking for a `.git` to establish the
    project root and otherwise can't see the files. A bare `git init` provides that marker. jj
    always ignores `.git`, so it never appears in the collected diff.
    """
    if (workspace / ".git").exists():
        return
    subprocess.run(["git", "init", "-q"], cwd=workspace, capture_output=True, text=True)
    # Stage the tree so tools that enumerate project files via `git ls-files` (opencode) see
    # them; a bare `git init` leaves an empty index. jj ignores `.git`, so none of this is in
    # the collected diff.
    subprocess.run(["git", "add", "-A"], cwd=workspace, capture_output=True, text=True)


def _diff_exclude_fileset() -> list[str]:
    """A jj fileset arg (or none) dropping run cruft (open-mem cache, bytecode) from diffs."""
    if not _DIFF_EXCLUDE_FILESETS:
        return []
    return [" & ".join(f"~{term}" for term in _DIFF_EXCLUDE_FILESETS)]


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


def collect_diff(workspace: Path, base_rev: str) -> tuple[str, DiffStat]:
    """Collect the unified diff and a parsed DiffStat for everything since base_rev."""
    fileset = _diff_exclude_fileset()
    diff_proc = _run_jj(
        ["diff", "--from", base_rev, "--git", *fileset],
        cwd=workspace,
    )
    stat_proc = _run_jj(
        ["diff", "--from", base_rev, "--stat", *fileset],
        cwd=workspace,
    )
    return diff_proc.stdout, _parse_diff_stat(stat_proc.stdout)


def forget_workspace(repo: Path, workspace: Path) -> None:
    """Tell jj to forget the workspace, then remove the directory."""
    name = workspace.name
    # check=False because we still want to rmtree even if forget fails
    _run_jj(["workspace", "forget", name], cwd=repo, check=False)
    if workspace.exists():
        shutil.rmtree(workspace, ignore_errors=True)
