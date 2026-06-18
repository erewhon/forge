"""Parallel Edit — run the same prompt against the same repo with N models, then compare."""

from __future__ import annotations

import argparse
import asyncio
import string
import sys
from datetime import UTC, datetime
from pathlib import Path

from agents.parallel_edit.config import settings
from agents.parallel_edit.judge import judge_runs
from agents.parallel_edit.logger import log_run
from agents.parallel_edit.models import CandidateSpec, ParallelEditResult
from agents.parallel_edit.renderer import render_markdown
from agents.parallel_edit.runner import cleanup_runs_selective, run_all
from agents.parallel_edit.workspaces import resolve_base_rev


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise SystemExit("error: provide --prompt TEXT, --prompt-file PATH, or pipe a prompt via stdin")


def _parse_one_candidate(raw: str, label: str) -> CandidateSpec:
    """Parse a "[kind:]model" token into a CandidateSpec.

    Bare value or "claude:<id>" -> claude. "opencode:<ref>" -> opencode, with the configured
    prefix (default "llm/") added when the ref has no provider segment.
    """
    kind, sep, rest = raw.partition(":")
    if sep and kind in ("claude", "opencode"):
        model = rest.strip()
    else:
        kind, model = "claude", raw.strip()

    if kind == "opencode":
        if "/" not in model:
            model = f"{settings.opencode_model_prefix}{model}"
        display = f"opencode:{model}"
    else:
        display = model
    return CandidateSpec(label=label, kind=kind, model=model, display=display)


_MAX_CANDIDATES = len(string.ascii_uppercase)  # labels A..Z


def _parse_candidates(raw: str | None) -> list[CandidateSpec]:
    tokens = settings.default_candidate_models if raw is None else raw.split(",")
    tokens = [t.strip() for t in tokens if t.strip()]
    if not 2 <= len(tokens) <= _MAX_CANDIDATES:
        raise SystemExit(
            f"error: --models must list between 2 and {_MAX_CANDIDATES} candidates, "
            f"got {len(tokens)}: {tokens}"
        )
    labels = list(string.ascii_uppercase[: len(tokens)])
    return [_parse_one_candidate(tok, label) for label, tok in zip(labels, tokens, strict=True)]


async def _run(args: argparse.Namespace) -> int:
    prompt = _read_prompt(args)
    if not prompt.strip():
        raise SystemExit("error: prompt is empty")

    candidates = _parse_candidates(args.models)
    repo = Path(args.repo).resolve()
    if not (repo / ".jj").exists():
        raise SystemExit(f"error: {repo} is not a jj repo (no .jj directory)")

    if args.keep_workspaces:
        settings.cleanup_on_success = False
        settings.cleanup_on_failure = False

    base_rev = resolve_base_rev(repo, args.base)

    roster = " vs ".join(f"{c.label}={c.display}" for c in candidates)
    print(
        f"Parallel edit @ {repo} (base {base_rev[:12]}): {roster}",
        file=sys.stderr,
    )

    runs = await run_all(prompt=prompt, candidates=candidates, repo=repo, base_rev=base_rev)

    for run in runs:
        line = f"  {run.label} ({run.model}): {run.status}"
        if run.latency_ms is not None:
            line += f" — {run.latency_ms} ms"
        if run.status in ("ok", "no_changes"):
            stat = run.diff_stat
            line += f" — {stat.files_changed}f / +{stat.insertions} / -{stat.deletions}"
        if run.error_message:
            line += f" — {run.error_message}"
        print(line, file=sys.stderr)

    print("Judging...", file=sys.stderr)
    verdict, judge_model, judge_error = await judge_runs(prompt=prompt, runs=runs)
    if judge_error:
        print(f"  judge: {judge_error}", file=sys.stderr)
    elif verdict is not None:
        print(f"  judge: winner={verdict.winner} (model={judge_model})", file=sys.stderr)

    result = ParallelEditResult(
        prompt=prompt,
        repo_path=repo,
        base_rev=base_rev,
        timestamp=datetime.now(UTC),
        runs=runs,
        verdict=verdict,
        judge_model=judge_model,
        judge_error=judge_error,
    )

    markdown = render_markdown(result)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        print(f"Report written to {out_path}", file=sys.stderr)
    else:
        sys.stdout.write(markdown)

    log_path = log_run(result)
    print(f"Run logged to {log_path}", file=sys.stderr)

    if args.keep_workspaces:
        kept = [r.workspace_path for r in runs if r.workspace_path.exists()]
    else:
        kept = cleanup_runs_selective(repo, runs)
    if kept:
        print("Workspaces kept for inspection:", file=sys.stderr)
        for ws in kept:
            print(f"  {ws}", file=sys.stderr)

    if verdict is None:
        return 2
    failed_runs = [r for r in runs if r.status not in ("ok", "no_changes")]
    return 2 if failed_runs else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the same prompt against a jj repo with N models (2–26), then compare.",
    )
    parser.add_argument("--prompt", default=None, help="The prompt text (inline)")
    parser.add_argument("--prompt-file", default=None, help="Path to a file containing the prompt")
    parser.add_argument(
        "--models",
        default=None,
        help="2–26 comma-separated candidates as '[kind:]model'. Bare or 'claude:<id>' runs "
        "claude -p; 'opencode:<ref>' runs opencode against the router (e.g. "
        "'claude-opus-4-8,opencode:glm-5.1,opencode:qwen3.6-plus'). Defaults to "
        "PARALLEL_EDIT_DEFAULT_CANDIDATE_MODELS.",
    )
    parser.add_argument("--repo", default=".", help="Path to the jj repo to edit (default: cwd)")
    parser.add_argument(
        "--base",
        default="@",
        help="jj revset to use as the base for diffs (default: @ at invocation time)",
    )
    parser.add_argument(
        "--output", default=None, help="Write the markdown report here (default: stdout)"
    )
    parser.add_argument(
        "--keep-workspaces",
        action="store_true",
        help="Keep all candidate workspaces on disk after the run",
    )

    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        # Best-effort: don't leave workspaces around if the user ctrl-c's
        # (cleanup is the caller's problem at this point — we don't know which runs exist)
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
