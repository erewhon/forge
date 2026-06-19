"""Testing-review ensemble — entry point. `meta testing <paths...> [--focus ...]`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agents.testing_ensemble.review import render, run_review


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
    args = parser.parse_args(argv)

    if args.emit_tasks and not args.project:
        print("error: --emit-tasks requires --project", file=sys.stderr)
        return 2

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
