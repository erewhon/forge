"""Dependency bumper entry point — `meta deps`.

Scan, bump, gate, and (when eligible + all gates pass) auto-merge clean patch/minor dependency
bumps.  Fail-closed: any gate miss or policy-ineligible candidate falls through to an advisory
branch plus a Forge task, never a merge.

Also supports a read-only redundancy report sub-mode: ``--redundancy-report`` asks a model which
dependencies overlap in purpose and prints a markdown report to stdout.

Usage::

    meta deps                   # run the loop on the current repo
    meta deps --dry-run         # plan only (no writes)
    meta deps --auto-merge      # also advance main when every gate passes
    meta deps --redundancy-report  # print a markdown redundancy report and exit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from forge.dependabot.autobump import auto_bump, render_bump
from forge.dependabot.redundancy import redundancy_report, render_report
from forge.shared.automerge import find_repo_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dependency bumper: scan, gate, and auto-merge clean low-risk bumps."
    )
    parser.add_argument(
        "--auto-merge",
        action="store_true",
        default=False,
        help="Advance main to the bump branch when every gate passes (default: OFF — fail-closed)",
    )
    parser.add_argument(
        "--project",
        default="Meta",
        help="Forge project name for advisory tasks (default: Meta)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Plan only: no writes, no gates, no bump application",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="Override the repo root path (default: auto-detect from cwd)",
    )
    parser.add_argument(
        "--redundancy-report",
        action="store_true",
        default=False,
        help="Print a read-only markdown report of overlapping-purpose dependency clusters",
    )
    args = parser.parse_args(argv)

    if args.repo:
        repo_path = Path(args.repo).expanduser().resolve()
    else:
        repo_path = find_repo_root(Path.cwd())
        if repo_path is None:
            print("error: no jj/git repo found in cwd or parents", file=sys.stderr)
            return 1

    # Read-only redundancy report: scan deps, ask the model, print markdown, exit 0.
    if args.redundancy_report:
        report, deps = redundancy_report(repo_path)
        print(render_report(report, deps))
        return 0

    result = auto_bump(
        repo_path,
        project=args.project,
        auto_merge=args.auto_merge,
        dry_run=args.dry_run,
    )

    print(render_bump(result))
    if result.status == "advisory" or result.status == "error":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
