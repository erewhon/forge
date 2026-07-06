from __future__ import annotations

import subprocess
from pathlib import Path

from agents.code_reviewer.config import settings
from agents.code_reviewer.models import RepoChanges


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command with standard options."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=cwd,
    )


def _truncate_diff(diff_text: str, max_lines: int) -> tuple[str, bool]:
    """Truncate diff to max_lines. Returns (text, was_truncated)."""
    lines = diff_text.splitlines()
    if len(lines) <= max_lines:
        return diff_text, False
    return "\n".join(lines[:max_lines]) + "\n... (truncated)", True


def _collect_jj(repo_path: Path, lookback_hours: int, max_lines: int) -> RepoChanges | None:
    """Collect recent changes from a jj repo."""
    revset = f'trunk()..@ & ~empty() & committer_date(after:"{lookback_hours} hours ago")'

    # Get commit list
    try:
        result = _run(
            [
                "jj",
                "log",
                "--no-pager",
                "-r",
                revset,
                "--no-graph",
                "-T",
                'change_id.short() ++ " " ++ description.first_line() ++ "\n"',
            ],
            cwd=repo_path,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  Warning: jj log failed for {repo_path.name}: {e}")
        return None

    if result.returncode != 0:
        print(f"  Warning: jj log failed for {repo_path.name}: {result.stderr.strip()}")
        return None

    commits = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    if not commits:
        return None

    # Get full diff
    try:
        diff_result = _run(
            ["jj", "diff", "--no-pager", "-r", revset, "--git"],
            cwd=repo_path,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  Warning: jj diff failed for {repo_path.name}: {e}")
        return None

    diff_text = diff_result.stdout if diff_result.returncode == 0 else ""

    # Get stat summary
    try:
        stat_result = _run(
            ["jj", "diff", "--no-pager", "-r", revset, "--stat"],
            cwd=repo_path,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  Warning: jj diff --stat failed for {repo_path.name}: {e}")
        stat_result = None

    diff_stat = stat_result.stdout.strip() if stat_result and stat_result.returncode == 0 else ""

    truncated_diff, was_truncated = _truncate_diff(diff_text, max_lines)

    return RepoChanges(
        repo_name=repo_path.name,
        vcs="jj",
        commit_count=len(commits),
        commit_summaries=commits,
        diff_stat=diff_stat,
        diff_text=truncated_diff,
        truncated=was_truncated,
    )


def _collect_git(repo_path: Path, lookback_hours: int, max_lines: int) -> RepoChanges | None:
    """Collect recent changes from a git repo."""
    since = f"{lookback_hours} hours ago"

    # Get commit list
    try:
        result = _run(
            ["git", "log", "--oneline", f"--since={since}"],
            cwd=repo_path,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  Warning: git log failed for {repo_path.name}: {e}")
        return None

    if result.returncode != 0:
        print(f"  Warning: git log failed for {repo_path.name}: {result.stderr.strip()}")
        return None

    commits = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    if not commits:
        return None

    # Get full diff
    try:
        diff_result = _run(
            ["git", "log", "-p", f"--since={since}"],
            cwd=repo_path,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  Warning: git log -p failed for {repo_path.name}: {e}")
        return None

    diff_text = diff_result.stdout if diff_result.returncode == 0 else ""

    # Get stat summary
    try:
        stat_result = _run(
            ["git", "log", "--stat", f"--since={since}"],
            cwd=repo_path,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  Warning: git log --stat failed for {repo_path.name}: {e}")
        stat_result = None

    diff_stat = stat_result.stdout.strip() if stat_result and stat_result.returncode == 0 else ""

    truncated_diff, was_truncated = _truncate_diff(diff_text, max_lines)

    return RepoChanges(
        repo_name=repo_path.name,
        vcs="git",
        commit_count=len(commits),
        commit_summaries=commits,
        diff_stat=diff_stat,
        diff_text=truncated_diff,
        truncated=was_truncated,
    )


def collect_all() -> list[RepoChanges]:
    """Iterate over configured repos and collect recent changes."""
    all_changes: list[RepoChanges] = []

    for repo_path in settings.repos_paths:
        if not repo_path.is_dir():
            print(f"  Skipping {repo_path.name}: directory not found")
            continue

        # Detect VCS type (prefer jj over git)
        if (repo_path / ".jj").is_dir():
            changes = _collect_jj(repo_path, settings.lookback_hours, settings.max_diff_lines)
        elif (repo_path / ".git").exists():
            changes = _collect_git(repo_path, settings.lookback_hours, settings.max_diff_lines)
        else:
            print(f"  Skipping {repo_path.name}: no VCS detected")
            continue

        if changes is not None:
            all_changes.append(changes)
            truncated_note = " (truncated)" if changes.truncated else ""
            print(
                f"  {changes.repo_name}: {changes.commit_count} commits "
                f"via {changes.vcs}{truncated_note}"
            )

    return all_changes
