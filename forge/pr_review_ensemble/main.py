"""PR Review Ensemble — fan out a diff to multiple LLM providers, synthesize an advisory."""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

from agents.pr_review_ensemble.digest import run_digest
from agents.pr_review_ensemble.logger import log_digest, log_run, log_supply_chain
from agents.pr_review_ensemble.renderer import render_digest, render_markdown, render_supply_chain
from agents.pr_review_ensemble.runner import run_ensemble
from agents.pr_review_ensemble.supply_chain import run_supply_chain_audit


def _gh(args: list[str]) -> str:
    """Run a `gh` command, returning stdout; raises SystemExit with stderr on failure."""
    proc = subprocess.run(["gh", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"error: `gh {' '.join(args)}` failed: {proc.stderr.strip()}")
    return proc.stdout


def _fetch_pr_diff(pr: int, repo: str | None) -> str:
    """The unified diff of a GitHub PR via `gh pr diff` — the work-deployable input path."""
    cmd = ["pr", "diff", str(pr)]
    if repo:
        cmd += ["--repo", repo]
    return _gh(cmd)


def _post_pr_comment(pr: int, repo: str | None, body: str) -> None:
    cmd = ["pr", "comment", str(pr), "--body", body]
    if repo:
        cmd += ["--repo", repo]
    _gh(cmd)


def _read_diff(args: argparse.Namespace) -> str:
    if args.pr is not None:
        return _fetch_pr_diff(args.pr, args.repo)
    if args.diff_file:
        return Path(args.diff_file).read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("error: provide --pr N, --diff-file PATH, or pipe a diff via stdin")


def _maybe_post(markdown: str, args: argparse.Namespace, *, label: str) -> None:
    """Post the advisory back to the PR when --post-comment is set (requires --pr)."""
    if args.post_comment and args.pr is not None:
        _post_pr_comment(args.pr, args.repo, markdown)
        print(f"{label} posted to PR #{args.pr}", file=sys.stderr)


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
    markdown = render_markdown(result)
    _emit(markdown, args, label="Advisory")
    _maybe_post(markdown, args, label="Advisory")
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
    markdown = render_digest(result)
    _emit(markdown, args, label="Digest")
    _maybe_post(markdown, args, label="Digest")
    print(f"Run logged to {log_digest(result)}", file=sys.stderr)
    if result.digest is not None:
        print(f"Digest by {result.model}", file=sys.stderr)
        return 0
    print(f"Digest not produced: {result.error}", file=sys.stderr)
    return 2


async def _run_supply_chain(diff_text: str, pr_ref: str, args: argparse.Namespace) -> int:
    print(f"Running supply-chain audit on {pr_ref}...", file=sys.stderr)
    result = await run_supply_chain_audit(diff_text=diff_text, pr_ref=pr_ref)
    markdown = render_supply_chain(result)
    _emit(markdown, args, label="Supply-chain audit")
    _maybe_post(markdown, args, label="Supply-chain audit")
    print(f"Run logged to {log_supply_chain(result)}", file=sys.stderr)
    scan = result.scan
    print(
        f"Pre-scan: {len(scan.signals)} signal(s) in {len(scan.relevant_files)} file(s)",
        file=sys.stderr,
    )
    if result.ensemble is None:
        print("Verdict: CLEAR (no supply-chain surface)", file=sys.stderr)
        return 0
    if result.ensemble.quorum_state == "failed":
        return 2
    return 0


def _default_pr_ref(args: argparse.Namespace) -> str:
    if args.pr is not None:
        return f"{args.repo}#{args.pr}" if args.repo else f"#{args.pr}"
    return "(unspecified)"


async def _run(args: argparse.Namespace) -> int:
    if args.post_comment and args.pr is None:
        raise SystemExit("error: --post-comment requires --pr")
    diff_text = _read_diff(args)
    if not diff_text.strip():
        raise SystemExit("error: diff is empty")

    pr_ref = args.pr_ref or _default_pr_ref(args)
    if args.pass_ == "digest":
        return await _run_digest(diff_text, pr_ref, args)
    if args.pass_ == "supply-chain":
        return await _run_supply_chain(diff_text, pr_ref, args)
    return await _run_review(diff_text, pr_ref, args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PR Review Ensemble (advisory)")
    parser.add_argument(
        "--pass",
        dest="pass_",
        choices=["review", "digest", "supply-chain"],
        default="review",
        help="Which lens to run: 'review' (fan-out + synthesize advisory), 'digest' (navigational "
        "digest of a large PR), or 'supply-chain' (deterministic pre-scan + focused audit of "
        "dependency/hook/CI/obfuscation changes). Default: review.",
    )
    parser.add_argument(
        "--pr",
        type=int,
        default=None,
        help="GitHub PR number to review; fetches the diff via `gh pr diff` "
        "(the work-deployable input path). Overrides --diff-file/stdin.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="owner/repo for --pr and --post-comment (default: gh's current repo).",
    )
    parser.add_argument(
        "--post-comment",
        action="store_true",
        help="Post the advisory back to the PR as a comment via `gh pr comment` (requires --pr).",
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
        help="Identifier for this PR (used in output and logs); defaults from --pr/--repo",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write the advisory markdown (default: stdout)",
    )
    args = parser.parse_args(argv)

    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
