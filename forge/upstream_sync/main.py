"""Upstream sync entry point — `forge upstream`.

Fetch the fork's upstream, merge it on a sync branch inside a disposable worktree, gate
(green suite + collision seat), and push the branch. Fail-closed: a textual conflict or a
gate miss files an advisory task, never a merge; ``--auto-merge`` advances the remote
default branch only when every gate affirmatively passes.

Usage::

    forge upstream                    # sync the current repo's upstream remote
    forge upstream --dry-run          # compare only (no worktree, no writes)
    forge upstream --auto-merge       # also advance the default branch when all green
    forge upstream --repo <path>      # target another repo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from forge.shared.automerge import find_repo_root
from forge.upstream_sync.models import SyncResult
from forge.upstream_sync.sync import sync_upstream


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Upstream sync for additive forks: fetch, merge, gate, push."
    )
    parser.add_argument(
        "--auto-merge",
        action="store_true",
        default=False,
        help="Advance the default branch when every gate passes (default: OFF — fail-closed)",
    )
    parser.add_argument(
        "--project",
        default="Meta",
        help="Task-store project for advisory tasks (default: Meta)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Compare only: no worktree, no merge, no writes",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="Override the repo root path (default: auto-detect from cwd)",
    )
    args = parser.parse_args(argv)

    if args.repo:
        repo_path = Path(args.repo).expanduser().resolve()
    else:
        repo_path = find_repo_root(Path.cwd())
        if repo_path is None:
            print("error: no jj/git repo found in cwd or parents", file=sys.stderr)
            return 1

    result = sync_upstream(
        repo_path,
        project=args.project,
        auto_merge=args.auto_merge,
        dry_run=args.dry_run,
    )

    print(render_sync(result))
    if result.status in ("conflict", "advisory", "error"):
        return 1
    return 0


def render_sync(result: SyncResult) -> str:
    """One-screen human summary of a sync run."""
    lines = [f"# forge upstream — {result.status}"]
    if result.reason:
        lines += ["", result.reason]
    if result.commits_behind:
        lines += [
            "",
            f"- Upstream: {result.commits_behind} commit(s) since merge-base "
            f"(tip `{(result.upstream_tip or '')[:8]}`)",
        ]
    if result.layer is not None:
        lines.append(
            f"- Layer: {len(result.layer.added)} fork-added / "
            f"{len(result.layer.modified)} fork-modified; "
            f"overlap {len(result.overlap)} file(s)"
        )
    if result.branch:
        lines.append(f"- Branch: {result.branch}")
    if result.conflicted:
        lines.append(f"- Conflicts: {', '.join(result.conflicted[:10])}")
    if result.tests_passed is not None:
        lines.append(f"- Suite: {'green' if result.tests_passed else 'RED'}")
    if result.collision is not None:
        state = {True: "COLLISION", False: "clear", None: "unknown"}[result.collision.collision]
        lines.append(f"- Collision seat: {state}")
    if result.merged_to_main:
        lines.append("- Merged to the default branch (local copy is now behind — pull)")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
