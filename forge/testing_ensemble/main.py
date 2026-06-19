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
    args = parser.parse_args(argv)

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
