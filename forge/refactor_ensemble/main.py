"""Refactoring ensemble — entry point. `meta refactor <paths...> [--focus ...]`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agents.refactor_ensemble.plan import render, run_refactor


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
    args = parser.parse_args(argv)

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
