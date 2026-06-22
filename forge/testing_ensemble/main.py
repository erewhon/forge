"""Testing-review ensemble — entry point. `meta testing <paths...> [--focus ...]`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agents.testing_ensemble.review import render, run_review


def _run_auto(args: argparse.Namespace) -> int:
    """`meta testing --auto ...` — the generate → gate → push/merge loop."""
    from agents.shared.automerge import find_repo_root
    from agents.testing_ensemble.autotest import auto_test, render_auto

    if args.repo:
        repo_path = Path(args.repo).expanduser().resolve()
    else:
        repo_path = find_repo_root(Path(args.paths[0])) or Path.cwd()

    try:
        result = auto_test(
            args.paths,
            repo_path=repo_path,
            focus=args.focus,
            project=args.project,
            auto_merge=args.auto_merge,
            max_gaps=args.max_gaps,
            min_severity=args.min_severity,
            branch_prefix=args.branch_prefix,
            dry_run=args.dry_run,
            log=lambda m: print(f"  auto: {m}", file=sys.stderr),
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.output:
        Path(args.output).write_text(render_auto(result) + "\n")
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(render_auto(result) + "\n")
    return 1 if result.status == "error" else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Adversarial multi-model test-coverage review (discover → dedup → verify)"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Source and/or test files or dirs (tests are auto-detected; point at a module dir)",
    )
    parser.add_argument(
        "--focus",
        default="test coverage and robustness",
        help="What to aim the review at (e.g. 'concurrency', 'error paths'). Default: coverage.",
    )
    parser.add_argument("--output", default=None, help="Write the report here (default: stdout)")
    parser.add_argument(
        "--emit-tasks",
        action="store_true",
        help="Emit confirmed gaps as Forge test tasks (review-then-implement) besides the report",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Forge project to file emitted tasks into (required with --emit-tasks; must exist)",
    )
    parser.add_argument(
        "--min-severity",
        choices=("critical", "high", "medium", "low"),
        default="low",
        help="Only emit confirmed gaps at or above this severity (default: low — emit all)",
    )
    parser.add_argument(
        "--dry-run-emit",
        action="store_true",
        help="With --emit-tasks: print what would be created, create nothing",
    )
    auto = parser.add_argument_group("auto-merge (--auto): generate tests, gate, and push a branch")
    auto.add_argument(
        "--auto",
        action="store_true",
        help="Generate tests for confirmed gaps, gate (tests-only + green + full-quorum sign-off), "
        "and push a branch. Blocked gates revert and fall back to --emit-tasks.",
    )
    auto.add_argument(
        "--auto-merge",
        action="store_true",
        help="With --auto: also advance `main` to the branch when every gate passes (loaded).",
    )
    auto.add_argument(
        "--repo",
        default=None,
        help="Repo root for VCS actions (default: derived from the first path, else cwd)",
    )
    auto.add_argument(
        "--max-gaps",
        type=int,
        default=None,
        help="Cap generated tests per --auto run (default: TESTING_ENSEMBLE_auto_max_gaps)",
    )
    auto.add_argument(
        "--branch-prefix",
        default="auto-tests",
        help="Branch/bookmark name prefix for --auto (default: auto-tests)",
    )
    auto.add_argument(
        "--dry-run",
        action="store_true",
        help="With --auto: plan only — no test generation, gating, or VCS writes",
    )
    args = parser.parse_args(argv)

    if args.emit_tasks and not args.project:
        print("error: --emit-tasks requires --project", file=sys.stderr)
        return 2
    if args.auto_merge and not args.auto:
        print("error: --auto-merge requires --auto", file=sys.stderr)
        return 2
    if args.auto:
        return _run_auto(args)

    try:
        report = run_review(args.paths, args.focus)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    markdown = render(report)
    if args.output:
        Path(args.output).write_text(markdown)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(markdown + "\n")
    print(
        f"Testing review: {len(report.confirmed)} confirmed, {len(report.tentative)} tentative "
        f"({report.raw_count} raw → {report.canonical_count} canonical)",
        file=sys.stderr,
    )

    if args.emit_tasks:
        from agents.testing_ensemble.emit import emit_report

        try:
            summary = emit_report(
                report,
                project=args.project,
                min_severity=args.min_severity,
                dry_run=args.dry_run_emit,
                log=lambda m: print(f"  emit: {m}", file=sys.stderr),
            )
        except ValueError as e:  # e.g. project folder doesn't exist
            print(f"error: task emission failed: {e}", file=sys.stderr)
            return 1
        prefix = "[dry-run] " if args.dry_run_emit else ""
        print(f"{prefix}{summary.line()}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
