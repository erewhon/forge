"""PR Review Ensemble — fan out a diff to multiple LLM providers, synthesize an advisory."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from agents.pr_review_ensemble.digest import run_digest
from agents.pr_review_ensemble.logger import log_digest, log_run
from agents.pr_review_ensemble.renderer import render_digest, render_markdown
from agents.pr_review_ensemble.runner import run_ensemble


def _read_diff(args: argparse.Namespace) -> str:
    if args.diff_file:
        return Path(args.diff_file).read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("error: provide --diff-file PATH or pipe a diff via stdin")


def _emit(markdown: str, args: argparse.Namespace, *, label: str) -> None:
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        print(f"{label} written to {out_path}", file=sys.stderr)
    else:
        sys.stdout.write(markdown)


async def _run_review(diff_text: str, pr_ref: str, args: argparse.Namespace) -> int:
    diff_lines = diff_text.count("\n") + 1
    print(f"Running review ensemble on {pr_ref} ({diff_lines} lines)...", file=sys.stderr)
    result = await run_ensemble(diff_text=diff_text, pr_ref=pr_ref)
    _emit(render_markdown(result), args, label="Advisory")
    print(f"Run logged to {log_run(result)}", file=sys.stderr)
    print(
        f"Quorum: {result.quorum_state} "
        f"({len(result.providers_succeeded)}/{len(result.providers_attempted)})",
        file=sys.stderr,
    )
    return 2 if result.quorum_state == "failed" else 0


async def _run_digest(diff_text: str, pr_ref: str, args: argparse.Namespace) -> int:
    diff_lines = diff_text.count("\n") + 1
    print(f"Running digest on {pr_ref} ({diff_lines} lines)...", file=sys.stderr)
    result = await run_digest(diff_text=diff_text, pr_ref=pr_ref)
    _emit(render_digest(result), args, label="Digest")
    print(f"Run logged to {log_digest(result)}", file=sys.stderr)
    if result.digest is not None:
        print(f"Digest by {result.model}", file=sys.stderr)
        return 0
    print(f"Digest not produced: {result.error}", file=sys.stderr)
    return 2


async def _run(args: argparse.Namespace) -> int:
    diff_text = _read_diff(args)
    if not diff_text.strip():
        raise SystemExit("error: diff is empty")

    pr_ref = args.pr_ref or "(unspecified)"
    if args.pass_ == "digest":
        return await _run_digest(diff_text, pr_ref, args)
    return await _run_review(diff_text, pr_ref, args)


def main() -> None:
    parser = argparse.ArgumentParser(description="PR Review Ensemble (advisory)")
    parser.add_argument(
        "--pass",
        dest="pass_",
        choices=["review", "digest"],
        default="review",
        help="Which lens to run: 'review' (fan-out + synthesize advisory) or 'digest' "
        "(single resilient navigational digest of a large PR). Default: review.",
    )
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
