"""Backlog queue report — `forge queue`.

Read-only: prints every project's non-done tasks grouped by project, with the worker's
dispatch gate resolved per row. ``--auto`` narrows to Auto-OK / Auto-Preferred tasks (and
drops projects without any), which answers "where can I point the task worker right now".

Usage::

    forge queue                     # all projects, all non-done tasks
    forge queue --auto              # only auto-mode tasks, only projects that have them
    forge queue --project Meta      # one project
"""

from __future__ import annotations

import argparse

from forge.queue.models import QueueRow
from forge.shared.task_store import get_task_store

# Blocked rows list their blockers inline; past this many, the rest collapse to a count.
_MAX_BLOCKERS_SHOWN = 2


def render_queue(rows: list[QueueRow], *, auto_only: bool = False) -> str:
    """The grouped report. Projects alphabetical; rows by priority then title."""
    if auto_only:
        rows = [r for r in rows if r.is_auto]
    if not rows:
        return "No auto-mode tasks open." if auto_only else "No open tasks."

    by_project: dict[str, list[QueueRow]] = {}
    for row in rows:
        by_project.setdefault(row.project or "(no project)", []).append(row)

    lines: list[str] = []
    for project in sorted(by_project, key=str.lower):
        project_rows = sorted(by_project[project], key=lambda r: (r.priority, r.task.lower()))
        ready = sum(1 for r in project_rows if r.is_dispatchable)
        # Under --auto the group holds only auto-mode rows, so "open" would undercount.
        noun = "auto-mode" if auto_only else "open"
        lines.append(f"{project} — {len(project_rows)} {noun}, {ready} auto-ready")
        for r in project_rows:
            mode = r.execution_mode + (f":{r.model_tier}" if r.is_auto and r.model_tier else "")
            line = f"  p{r.priority:<3} {r.status:<12} {mode:<15} {r.task}"
            if r.feature:
                line += f"  ({r.feature})"
            if r.blocked:
                shown = r.blocked_by[:_MAX_BLOCKERS_SHOWN]
                more = len(r.blocked_by) - len(shown)
                suffix = ", ".join(shown) + (f" +{more} more" if more > 0 else "")
                line += f"  [blocked by: {suffix}]"
            lines.append(line)
        lines.append("")
    return "\n".join(lines).rstrip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backlog report: non-done tasks per project, worker-readiness resolved."
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Only this project (default: every project in the task store)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        default=False,
        help="Only Auto-OK / Auto-Preferred tasks, only projects that have them",
    )
    args = parser.parse_args(argv)

    rows = get_task_store().queue(project=args.project)
    print(render_queue(rows, auto_only=args.auto))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
