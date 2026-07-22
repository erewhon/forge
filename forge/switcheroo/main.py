"""``forge switcheroo`` — manual-gated outage failover.

Usage::

    forge switcheroo now                       # drain worker-ready leaves via the local fleet
    forge switcheroo now --reason "all agents 529"   # annotate why (goes in the journal)
    forge switcheroo now --goal "..."          # synthesize a baton if none exists on disk
    forge switcheroo now --project Nous --max 5 --dry-run
    forge switcheroo back                       # resume briefing; re-anchor baton; archive window
    forge switcheroo back --dry-run             # briefing only, mutate nothing
    forge switcheroo status                     # show the baton + active failover window
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from forge.shared.baton import Baton, read_baton, write_baton
from forge.shared.task_store import get_task_store
from forge.switcheroo.drain import drain, worker_ready_rows
from forge.switcheroo.journal import (
    archive_failover,
    end_failover,
    read_failover,
    record_outcome,
    render_failover_summary,
    start_failover,
)
from forge.switcheroo.reconcile import home_repo_changes, render_switchback
from forge.task_worker.config import settings

_FAILOVER_NEXT_ACTION = "Failover: drain worker-ready Forge leaves via the local fleet."
_RECONCILE_NEXT_ACTION = (
    "Reconcile the failover window (see .forge/switcheroo), then continue the plan."
)


def _ensure_baton(home: Path, goal: str | None) -> Baton | None:
    """Read the session's baton (the handoff the interactive session should have left), or
    synthesize a minimal one from *goal*. Returns ``None`` — the caller aborts — when there is
    neither, because a failover with no record of where we were is exactly what the baton exists to
    prevent. The write re-anchors the baton to the current working copy: the failover-start line."""
    baton = read_baton(home)
    if baton is None:
        if not goal:
            return None
        baton = Baton(
            goal=goal,
            next_action=_FAILOVER_NEXT_ACTION,
            notes="Baton auto-created by `forge switcheroo now` (no prior handoff on disk).",
        )
    elif goal:
        baton = baton.model_copy(update={"goal": goal})
    return write_baton(home, baton)


def switcheroo_now(
    *,
    goal: str | None = None,
    reason: str = "",
    projects: list[str] | None = None,
    max_leaves: int = 0,
    dry_run: bool = False,
    model_tier: str | None = None,
    home: Path | None = None,
    store: object | None = None,
    run_one_fn: object | None = None,
) -> int:
    """Run one failover window. ``store``/``run_one_fn`` are injectable for tests; production leaves
    them ``None`` to resolve the real Nous task store and the OpenCode-backed worker."""
    home = home or Path.cwd()

    baton = _ensure_baton(home, goal)
    if baton is None:
        print(
            "error: no baton at .forge/baton.md — nothing to hand off. The interactive session "
            "should write one (it records where you were), or pass --goal to synthesize a minimal "
            "one for this failover.",
            file=sys.stderr,
        )
        return 1

    tier = model_tier or settings.model_tier_default
    if model_tier:
        settings.model_tier_default = model_tier  # so auto-tier leaves run under the chosen tier
    if dry_run:
        settings.dry_run = True  # belt-and-suspenders; we return before draining anyway

    store = store or get_task_store()
    allowed = list(projects) if projects else list(settings.allowed_projects)
    ready = worker_ready_rows(store, allowed)

    scope = ", ".join(allowed) if allowed else "all projects"
    print(f"switcheroo — home: {home}")
    print(f"  baton goal:  {baton.goal or '(none)'}")
    print(f"  anchor:      {baton.vcs or 'vcs'} {baton.change_id or '(unversioned)'}")
    print(f"  fleet tier:  {tier}   scope: {scope}")
    print(f"  worker-ready leaves: {len(ready)}" + (f"  (cap {max_leaves})" if max_leaves else ""))
    for r in ready[: max_leaves or len(ready)]:
        print(f"    - [{r.execution_mode} p{r.priority}] {r.project} / {r.task}")

    if dry_run:
        print("\n[dry run] baton re-anchored; running no leaves.")
        return 0
    if not ready:
        print("\nNothing worker-ready to drain. Baton re-anchored; no failover window opened.")
        return 0

    print(f"\nDraining under tier '{tier}'…\n")
    start_failover(home, baton=baton, model_tier=tier, reason=reason)

    def _on_outcome(leaf) -> None:
        record_outcome(home, leaf)
        tail = f" — {leaf.reason}" if leaf.reason else ""
        print(f"  [{leaf.status:>7}] {leaf.project} / {leaf.task}{tail}")

    drain(
        store=store,
        allowed=allowed,
        max_leaves=max_leaves,
        run_one_fn=run_one_fn,
        on_outcome=_on_outcome,
    )
    log = end_failover(home)

    print("\n" + "=" * 60)
    if log is not None:
        print(render_failover_summary(log))
    return 0


def switch_back(*, home: Path | None = None, dry_run: bool = False) -> int:
    """Return control after a failover window: render the resume briefing (baton + journal +
    home-repo diff), then — unless *dry_run* — re-anchor the baton to the post-failover state and
    archive the consumed window. Read-mostly by design; the real reconciliation is the human/Claude
    reading the briefing and folding the fleet's commits back in."""
    home = home or Path.cwd()
    baton = read_baton(home)
    log = read_failover(home)

    if baton is None and log is None:
        print("Nothing to switch back from: no baton and no failover window.")
        return 0

    was_open = log is not None and log.ended_at is None
    if log is not None and was_open and not dry_run:
        # The window never closed — the fleet (or the machine) was interrupted. Close it as part of
        # coming back so the briefing and archive reflect a finished window.
        log = end_failover(home) or log

    anchor = (log.baton_change_id if log else None) or (baton.change_id if baton else None)
    changes = home_repo_changes(home, anchor)

    print(render_switchback(baton, log, changes))
    if was_open:
        print("\n! The failover window was still open (interrupted) — reconcile carefully.")

    if dry_run:
        print("\n[dry run] briefing only; baton not re-anchored, window not archived.")
        return 0

    if baton is not None:
        # Re-anchor to the post-failover working copy and point next action at reconciliation.
        # Clearing the anchor fields makes write_baton recapture; decisions accrete (drift-safe).
        note = (
            f"Failover window {log.started_at}: fleet drained {len(log.outcomes)} leaf(s) "
            f"({len(log.done)} landed). Reconciled on switch-back."
            if log is not None
            else "Switched back from a failover window."
        )
        updated = baton.model_copy(
            update={
                "vcs": None,
                "branch": None,
                "change_id": None,
                "next_action": _RECONCILE_NEXT_ACTION,
                "decisions": [*baton.decisions, note],
            }
        )
        write_baton(home, updated)
    if log is not None:
        archive_failover(home)
        print("\nWindow archived to .forge/switcheroo/history/; baton re-anchored.")
    return 0


def _cmd_now(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="forge switcheroo now",
        description="Manual outage failover: drain worker-ready Forge leaves via the local fleet.",
    )
    parser.add_argument(
        "--goal", default=None, help="Synthesize a baton with this goal if none exists"
    )
    parser.add_argument(
        "--reason", default="", help="Why you're failing over (recorded in the journal)"
    )
    parser.add_argument(
        "--project", action="append", default=None, help="Limit to this project (repeatable)"
    )
    parser.add_argument("--max", type=int, default=0, help="Cap leaves drained (0 = until dry)")
    parser.add_argument(
        "--model-tier", default=None, help="Override the default fleet tier for the window"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show the plan; re-anchor the baton; run nothing"
    )
    args = parser.parse_args(argv)
    return switcheroo_now(
        goal=args.goal,
        reason=args.reason,
        projects=args.project,
        max_leaves=args.max,
        dry_run=args.dry_run,
        model_tier=args.model_tier,
    )


def _cmd_back(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="forge switcheroo back",
        description="Return control after a failover: resume briefing, re-anchor baton, archive.",
    )
    parser.add_argument("--home", type=Path, default=None, help="Home repo (default: cwd)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the briefing only; mutate nothing"
    )
    args = parser.parse_args(argv)
    return switch_back(home=args.home, dry_run=args.dry_run)


def _cmd_status(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="forge switcheroo status", description="Show the baton and the active failover window."
    )
    parser.add_argument("--home", type=Path, default=None, help="Home repo (default: cwd)")
    args = parser.parse_args(argv)
    home = args.home or Path.cwd()

    baton = read_baton(home)
    if baton is None:
        print(f"No baton at {home / '.forge' / 'baton.md'}.")
    else:
        print(f"Baton ({home / '.forge' / 'baton.md'}):")
        print(f"  goal:        {baton.goal or '(none)'}")
        print(f"  next action: {baton.next_action or '(none)'}")
        print(f"  anchor:      {baton.vcs or 'vcs'} {baton.change_id or '(unversioned)'}")
        print(f"  updated:     {baton.updated_at or '(unknown)'}")

    log = read_failover(home)
    print()
    if log is None:
        print("No active failover window.")
    else:
        print(render_failover_summary(log))
    return 0


_COMMANDS = {"now": _cmd_now, "back": _cmd_back, "status": _cmd_status}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:]) if argv is None else list(argv)
    if args and args[0] in _COMMANDS:
        return _COMMANDS[args[0]](args[1:])

    parser = argparse.ArgumentParser(
        prog="forge switcheroo",
        description="Manual-gated Claude-outage failover: drain worker-ready leaves via the fleet.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("now", help="Drain worker-ready leaves under the local fleet.")
    sub.add_parser("back", help="Return control: resume briefing, re-anchor baton, archive window.")
    sub.add_parser("status", help="Show the baton and the active failover window.")
    parser.parse_args(args)  # --help exits 0; unknown command exits 2
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
