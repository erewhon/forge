"""`forge grind` — iterate a goal via a runbook loop (reset → run → check → adjust), no commits.

Usage::

    forge grind ./grind.yaml                 # run the loop
    forge grind ./grind.yaml --dry-run       # validate + print the resolved plan, run nothing
    forge grind ./grind.yaml --model opencode/anthropic/claude-sonnet-4 --max-iterations 10
    forge grind init [path]                  # write a skeleton runbook to fill in
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from forge.grind.config import load_config, resolve_model, settings
from forge.grind.loop import grind
from forge.grind.models import GrindConfig
from forge.grind.scaffold import DEFAULT_FILENAME, write_skeleton

# Non-zero exits carry the loop's terminal status for scripting (0 = goal met / already done).
_EXIT = {"already-done": 0, "done": 0, "stuck": 2, "exhausted": 3, "blocked": 4}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "grind"


def _print_plan(cfg: GrindConfig, model: str, max_iterations: int, repo: Path) -> None:
    print(f"repo:  {repo}")
    print(f"model: {model}")
    print(f"goal:  {cfg.goal.strip()}")
    print(f"bound: {max_iterations} iterations, {cfg.no_progress_window}-turn no-progress guard")
    hill = "on" if cfg.check.score_regex else "off (add check.score_regex to enable)"
    print(f"hill-climb: {hill}")
    print("cycle:")
    for s in cfg.steps:
        print(f"  - {s.name}: {s.run}")
    print(f"  = check: {cfg.check.run}")
    print(f"observe: {', '.join(cfg.resolved_observe())}")
    if cfg.edit_paths:
        print(f"edit scope: {', '.join(cfg.edit_paths)}")


def run(
    config_path: str, *, cli_model: str | None, max_iterations: int | None, dry_run: bool
) -> int:
    cfg = load_config(config_path)
    repo = Path.cwd()
    model = resolve_model(cfg, cli_model)
    bound = max_iterations if max_iterations is not None else cfg.max_iterations
    if max_iterations is not None:
        cfg = cfg.model_copy(update={"max_iterations": max_iterations})
    cfg.resolved_observe()  # validate observe names up front (raises with a clear message)

    _print_plan(cfg, model, bound, repo)
    if dry_run:
        print("\n[dry run] validated; running nothing.")
        return 0

    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = repo / ".forge" / "grind" / "runs" / f"{_slug(cfg.goal)}-{ts}"
    print(f"\nrun dir: {run_dir}\n")

    outcome = grind(cfg, repo, model=model, run_dir=run_dir)
    print(f"\n{'=' * 60}\n{outcome.status.upper()}: {outcome.summary}")
    if outcome.best_score is not None:
        print(f"best score: {outcome.best_score}")
    return _EXIT.get(outcome.status, 1)


def _init(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="forge grind init", description="Write a skeleton grind runbook to fill in"
    )
    parser.add_argument(
        "path", nargs="?", default=None, help=f"Where to write it (default: ./{DEFAULT_FILENAME})"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite if it exists")
    args = parser.parse_args(argv)
    try:
        target = write_skeleton(args.path, force=args.force)
    except FileExistsError as e:
        print(f"error: {e} already exists (use --force to overwrite)", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"error: could not write skeleton: {e}", file=sys.stderr)
        return 1
    print(f"Wrote grind runbook skeleton to {target}")
    print(f"Edit it, then run:  forge grind {target} --dry-run")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "init":
        return _init(argv[1:])

    parser = argparse.ArgumentParser(
        description="Iterate a goal via a runbook loop (reset -> run -> check -> adjust).",
        epilog="No commits. Subcommand: `init [path]` writes a skeleton runbook.",
    )
    parser.add_argument("config", help="Path to the grind runbook (YAML/JSON)")
    parser.add_argument(
        "--model",
        default=None,
        help=f"OpenCode model string (default: config.model or {settings.model})",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=None, help="Override the runbook's iteration cap"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print the resolved plan; run nothing"
    )
    args = parser.parse_args(argv)
    return run(
        args.config,
        cli_model=args.model,
        max_iterations=args.max_iterations,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
