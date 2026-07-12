"""Fleet sweep entry point — `forge sweep`.

Enumerate the repos on a Soft Serve instance, keep workdir clones current, and run the
per-repo agents (`deps` always; `upstream` where a fork's upstream URL is configured),
each in its own subprocess with the task store pointed at that clone.

Usage::

    forge sweep --dry-run             # enumerate + clone + rehearse every agent
    forge sweep                       # the real thing (agents stay fail-closed per repo)
    forge sweep --only 'me/fork*'     # restrict this run to matching repos
    forge sweep --auto-merge          # pass the flag through to the agents (default OFF)
"""

from __future__ import annotations

import argparse

from forge.sweep.config import settings
from forge.sweep.driver import sweep
from forge.sweep.models import SweepResult


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sweep a Soft Serve instance: run the fleet agents on every repo."
    )
    parser.add_argument(
        "--host", default=None, help="SSH destination of the instance (default: SWEEP_HOST)"
    )
    parser.add_argument(
        "--only",
        default=None,
        metavar="GLOB",
        help="Restrict this run to repos matching GLOB (overrides include)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Pass --dry-run through to every agent run",
    )
    parser.add_argument(
        "--auto-merge",
        action="store_true",
        default=False,
        help="Pass --auto-merge through to every agent run (default: OFF)",
    )
    args = parser.parse_args(argv)

    if args.host:
        settings.host = args.host
    if args.only:
        settings.include = [args.only]

    result, code = sweep(dry_run=args.dry_run, auto_merge=args.auto_merge)
    print(render_sweep(result))
    return code


def render_sweep(result: SweepResult) -> str:
    """One-screen human summary: a row per agent run, driver errors last."""
    lines = [f"# forge sweep — {result.host or 'no host'}"]
    lines.append(
        f"{len(result.repos)} repo(s) swept, {len(result.skipped)} filtered out, "
        f"{len(result.runs)} agent run(s), {len(result.errors)} repo error(s)"
    )
    if result.runs:
        lines.append("")
        width = max(len(r.repo) for r in result.runs)
        for r in result.runs:
            row = f"{r.repo:<{width}}  {r.agent:<8}  {r.status}"
            if r.detail:
                first = r.detail.strip().splitlines()[0][:80]
                row += f"  — {first}"
            lines.append(row)
    if result.errors:
        lines += ["", "Repo errors (sweep continued):"]
        lines += [f"- {e}" for e in result.errors]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
