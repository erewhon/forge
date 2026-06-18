"""Code-audit ensemble — entry point. `meta audit <paths...> --focus "..."`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agents.code_audit_ensemble.audit import render, run_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Adversarial multi-model code audit (discover → dedup → verify)"
    )
    parser.add_argument("paths", nargs="+", help="Files or directories to audit")
    parser.add_argument(
        "--focus",
        default="correctness and reliability",
        help="What to aim the audit at (e.g. 'data loss', 'concurrency'). Default: correctness.",
    )
    parser.add_argument("--output", default=None, help="Write the report here (default: stdout)")
    args = parser.parse_args(argv)

    try:
        report = run_audit(args.paths, args.focus)
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
        f"Audit: {len(report.confirmed)} confirmed, {len(report.tentative)} tentative "
        f"({report.raw_count} raw → {report.canonical_count} canonical)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
