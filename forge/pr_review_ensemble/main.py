"""PR Review Ensemble — fan out a diff to multiple LLM providers, synthesize an advisory."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from agents.pr_review_ensemble.logger import log_run
from agents.pr_review_ensemble.renderer import render_markdown
from agents.pr_review_ensemble.runner import run_ensemble


def _read_diff(args: argparse.Namespace) -> str:
    if args.diff_file:
        return Path(args.diff_file).read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("error: provide --diff-file PATH or pipe a diff via stdin")


async def _run(args: argparse.Namespace) -> int:
    diff_text = _read_diff(args)
    if not diff_text.strip():
        raise SystemExit("error: diff is empty")

    pr_ref = args.pr_ref or "(unspecified)"
    diff_lines = diff_text.count("\n") + 1
    print(f"Running ensemble on {pr_ref} ({diff_lines} lines)...", file=sys.stderr)

    result = await run_ensemble(diff_text=diff_text, pr_ref=pr_ref)

    markdown = render_markdown(result)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        print(f"Advisory written to {out_path}", file=sys.stderr)
    else:
        sys.stdout.write(markdown)

    log_path = log_run(result)
    print(f"Run logged to {log_path}", file=sys.stderr)
    print(
        f"Quorum: {result.quorum_state} "
        f"({len(result.providers_succeeded)}/{len(result.providers_attempted)})",
        file=sys.stderr,
    )

    return 2 if result.quorum_state == "failed" else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="PR Review Ensemble (advisory)")
    parser.add_argument(
        "--diff-file",
        type=str,
        default=None,
        help="Path to diff file (default: read from stdin)",
    )
    parser.add_argument(
        "--pr-ref",
        type=str,
        default=None,
        help="Identifier for this PR (used in output and logs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write the advisory markdown (default: stdout)",
    )
    args = parser.parse_args()

    sys.exit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
