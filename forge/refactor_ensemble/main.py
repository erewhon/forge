"""Refactoring ensemble — entry point. `meta refactor <paths...> [--focus ...]`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from forge.refactor_ensemble.plan import render, run_refactor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Adversarial multi-model refactoring review (discover → dedup → verify)"
    )
    parser.add_argument("paths", nargs="+", help="Files or directories to review")
    parser.add_argument(
        "--focus",
        default="maintainability and best practices",
        help="What to aim the review at (e.g. 'duplication'). Default: maintainability.",
    )
    parser.add_argument("--output", default=None, help="Write the plan here (default: stdout)")
    parser.add_argument(
        "--emit-tasks",
        action="store_true",
        help="Emit confirmed smells as Forge tasks (review-then-implement) in addition to the plan",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Forge project to file emitted tasks into (required with --emit-tasks; must exist)",
    )
    parser.add_argument(
        "--min-impact",
        choices=("high", "medium", "low"),
        default="low",
        help="Only emit confirmed smells at or above this impact (default: low — emit all)",
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
        plan = run_refactor(args.paths, args.focus)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    markdown = render(plan)
    if args.output:
        Path(args.output).write_text(markdown)
        print(f"Plan written to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(markdown + "\n")
    print(
        f"Refactoring plan: {len(plan.confirmed)} confirmed, {len(plan.tentative)} tentative "
        f"({plan.raw_count} raw → {plan.canonical_count} canonical)",
        file=sys.stderr,
    )

    if args.emit_tasks:
        from forge.refactor_ensemble.emit import emit_plan

        try:
            summary = emit_plan(
                plan,
                project=args.project,
                min_impact=args.min_impact,
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
