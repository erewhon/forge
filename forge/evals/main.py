"""CLI front door for the eval harness: ``meta evals run | baseline | compare``.

Each subcommand is a thin wrapper around :func:`run_scorecard` and
:mod:`agents.evals.report`.  ``main`` uses argparse subparsers so the CLI
can be composed without Typer.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from agents.evals.config import settings
from agents.evals.report import render_scorecard, write_scorecard
from agents.evals.runner import run_scorecard

VALID_STEPS: list[str] = [
    "replan",
    "decompose",
    "boundedness",
    "review-findings",
    "review-confirm",
    "testgap-find",
    "testgap-skeptic",
]

BASELINES_DIR = Path(__file__).resolve().parent / "baselines"
BASELINE_FILE = BASELINES_DIR / ".json"


def _common_parser() -> argparse.ArgumentParser:
    """Return a parser shared by run/baseline/compare."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default=settings.model,
        help="Model identifier (default: EVALS_MODEL or 'coder').",
    )
    p.add_argument(
        "--step",
        action="append",
        default=None,
        help=(
            f"Step key to include (repeatable). Valid: {', '.join(VALID_STEPS)}. "
            "If omitted, runs all steps."
        ),
    )
    p.add_argument(
        "--goldsets",
        type=Path,
        default=None,
        help="Path to goldsets directory (default: settings.goldsets_dir).",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Number of repeats per case (default: settings.repeats).",
    )
    return p


def _validate_steps(steps: list[str] | None) -> list[str] | None:
    """Validate step keys against the known set; return None if empty."""
    if not steps:
        return None
    bad = [s for s in steps if s not in VALID_STEPS]
    if bad:
        print(f"Unknown step(s): {', '.join(bad)}. Valid: {', '.join(VALID_STEPS)}")
        return 1  # type: ignore[return-value]
    return steps


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _cmd_run(argv: list[str]) -> int:
    """Run a scorecard and print the markdown report."""
    parser = _common_parser()
    args = parser.parse_args(argv)

    steps = _validate_steps(args.step)
    if isinstance(steps, int):
        return 1

    sc = run_scorecard(
        model=args.model,
        steps=steps,
        goldsets_root=args.goldsets,
        repeats=args.repeats,
    )

    output_dir = args.goldsets.parent if args.goldsets else settings.goldsets_dir.parent
    json_path = write_scorecard(sc, output_dir)
    print(render_scorecard(sc))
    print(f"\nScorecard written to: {json_path}")
    return 0


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------


def _cmd_baseline(argv: list[str]) -> int:
    """Run a scorecard and persist it as the golden baseline."""
    parser = _common_parser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing baseline file.",
    )
    args = parser.parse_args(argv)

    steps = _validate_steps(args.step)
    if isinstance(steps, int):
        return 1

    if BASELINE_FILE.exists() and not args.force:
        print(f"Baseline already exists at {BASELINE_FILE}. Pass --force to overwrite.")
        return 1

    sc = run_scorecard(
        model=args.model,
        steps=steps,
        goldsets_root=args.goldsets,
        repeats=args.repeats,
    )

    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_FILE.write_text(sc.model_dump_json(indent=2), encoding="utf-8")
    print(f"Baseline written to: {BASELINE_FILE}")
    print(render_scorecard(sc))
    return 0


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def _cmd_compare(argv: list[str]) -> int:
    """Run a fresh scorecard and compare against the saved baseline."""
    parser = _common_parser()
    args = parser.parse_args(argv)

    steps = _validate_steps(args.step)
    if isinstance(steps, int):
        return 1

    if not BASELINE_FILE.exists():
        print(f"No baseline found at {BASELINE_FILE}. Run `meta evals baseline` first.")
        return 2

    baseline_data = json.loads(BASELINE_FILE.read_text())

    sc = run_scorecard(
        model=args.model,
        steps=steps,
        goldsets_root=args.goldsets,
        repeats=args.repeats,
    )

    # Build per-step delta table
    baseline_steps = {s["step"]: s for s in baseline_data.get("steps", [])}
    fresh_steps = {s.step: s for s in sc.steps}

    lines: list[str] = []
    lines.append(f"# Compare: {sc.model} vs baseline")
    lines.append("")
    lines.append("| Step | Baseline Pass Rate | Fresh Pass Rate | Delta |")
    lines.append("|------|-------------------:|----------------:|------:|")

    all_steps = sorted(set(list(baseline_steps.keys()) + list(fresh_steps.keys())))

    for step_name in all_steps:
        bl = baseline_steps.get(step_name)
        fr = fresh_steps.get(step_name)

        if bl and fr:
            bl_rate = bl.get("pass_rate", 0.0)
            fr_rate = fr.get("pass_rate", 0.0)
            delta = fr_rate - bl_rate
            lines.append(f"| {step_name} | {bl_rate:.0%} | {fr_rate:.0%} | {delta:+.0%} |")
        elif fr:
            lines.append(f"| {step_name} | (new) | {fr.pass_rate:.0%} | (new) |")
        elif bl:
            lines.append(
                f"| {step_name} | {bl.get('pass_rate', 0.0):.0%} | (missing) | (missing) |"
            )

    lines.append("")

    # Also show holdout deltas where available
    has_holdout = False
    for step_name in all_steps:
        bl = baseline_steps.get(step_name)
        fr = fresh_steps.get(step_name)
        if bl and fr:
            bl_hr = bl.get("holdout_pass_rate")
            fr_hr = fr.get("holdout_pass_rate")
            if bl_hr is not None and fr_hr is not None:
                has_holdout = True
                break

    if has_holdout:
        lines.append("| Step | Baseline Holdout | Fresh Holdout | Delta |")
        lines.append("|------|-----------------:|--------------:|------:|")
        for step_name in all_steps:
            bl = baseline_steps.get(step_name)
            fr = fresh_steps.get(step_name)
            if bl and fr:
                bl_hr = bl.get("holdout_pass_rate")
                fr_hr = fr.get("holdout_pass_rate")
                if bl_hr is not None and fr_hr is not None:
                    delta = fr_hr - bl_hr
                    lines.append(f"| {step_name} | {bl_hr:.0%} | {fr_hr:.0%} | {delta:+.0%} |")
            elif fr and fr.holdout_pass_rate is not None:
                lines.append(f"| {step_name} | (new) | {fr.holdout_pass_rate:.0%} | (new) |")
        lines.append("")

    # Mark regressions
    for step_name in all_steps:
        bl = baseline_steps.get(step_name)
        fr = fresh_steps.get(step_name)
        if bl and fr:
            bl_rate = bl.get("pass_rate", 0.0)
            fr_rate = fr.get("pass_rate", 0.0)
            if fr_rate < bl_rate:
                lines.append(
                    f"**REGRESSION**: `{step_name}` dropped from {bl_rate:.0%} to {fr_rate:.0%}"
                )

    md = "\n".join(lines)
    print(md)
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

_COMMANDS: dict[str, Callable[[list[str]], int]] = {
    "run": _cmd_run,
    "baseline": _cmd_baseline,
    "compare": _cmd_compare,
}


def main(argv: list[str] | None = None) -> int:
    """Dispatch ``meta evals`` subcommands.

    Known subcommands route to their handler; unknown or missing commands
    print help and exit 0 (advisory).
    """
    args = list(sys.argv[1:]) if argv is None else list(argv)
    if args and args[0] in _COMMANDS:
        return _COMMANDS[args[0]](args[1:])

    parser = argparse.ArgumentParser(
        prog="meta evals",
        description="Judgment eval harness: score models against frozen gold sets.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run", help="Run a scorecard and print the report.")
    sub.add_parser("baseline", help="Run a scorecard and save as baseline.")
    sub.add_parser("compare", help="Compare fresh scorecard against baseline.")
    parser.parse_args(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
