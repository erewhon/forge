"""CLI front door for the coding pipeline: ``meta build plan | run | status``.

Each subcommand is a thin wrapper around the pipeline's architectural layers.  ``main``
uses argparse subparsers so the CLI can be composed without Typer — each subcommand
is effectively its own entry point.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from forge.coding_pipeline.architect import (
    ArchitectError,
    decompose,
    persist_framing,
    persist_tree,
    propose_framing,
    require_approved_framing,
)
from forge.coding_pipeline.config import settings
from forge.coding_pipeline.emit import emit_tree
from forge.coding_pipeline.inventory import collect_inventory, run_dir_for, write_inventory
from forge.coding_pipeline.models import FramingProposal, GoalSpec, TaskTree, WaveRecord
from forge.coding_pipeline.orchestrator import OrchestratorResult, run_epic

# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def _cmd_plan(argv: list[str]) -> int:
    """A0+A1 (and optionally A2+A3 with --approve)."""
    parser = argparse.ArgumentParser(
        prog="meta build plan",
        description="Run the architect: inventory, framing proposal, and optional decomposition.",
    )
    parser.add_argument(
        "spec",
        nargs="?",
        default=None,
        help="Path to a goal spec (.yaml, .yml, or .md with YAML frontmatter).",
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Forge project name (must already exist).",
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help=(
            "Approve the framing and proceed with decomposition + emission. "
            "Requires an existing approved framing.json — without it this is an error."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="If an approved framing.json already exists, overwrite it with a new proposal.",
    )
    parsed = parser.parse_args(argv)

    # --- load spec --------------------------------------------------------
    if parsed.spec:
        spec_path = Path(parsed.spec)
        if not spec_path.is_file():
            print(f"Spec file not found: {spec_path}")
            return 1
        goal = GoalSpec.load(spec_path)
    else:
        goal = GoalSpec(goal="", project=parsed.project)

    # Run directory: derived from the goal (set epic_slug in the spec so plan and
    # plan --approve resolve the same dir).
    run_dir = run_dir_for(goal)
    framing_path = run_dir / "framing.json"

    # --- --approve path: A2+A3 against an ALREADY-approved framing ----------
    # No framing proposal happens here — --approve is never a shortcut past the
    # human gate, and never burns an architect call.
    if parsed.approve:
        if not framing_path.is_file():
            print(
                f"--approve was passed but {framing_path} does not exist. "
                "Run without --approve first to produce the proposal."
            )
            return 1
        existing = FramingProposal.model_validate_json(framing_path.read_text())
        if not existing.approved:
            print(
                "--approve requires an already-approved framing.json. "
                "Read framing.md, then set ``approved: true`` in framing.json."
            )
            return 1
        if existing.epic_slug != run_dir.name:
            print(
                f"note: framing epic_slug '{existing.epic_slug}' differs from the run dir "
                f"'{run_dir.name}' — refs will use the framing's slug"
            )

        print("[A0] Collecting inventory for decomposition context…")
        inv = collect_inventory(goal, repo=goal.repo or Path.cwd())

        print("[A2] Decomposing tree…")
        try:
            leaves = decompose(existing, inv)
        except ArchitectError as e:
            print(f"Decomposition failed: {e}")
            return 1

        print("[A3] Emitting task tree to Forge…")
        try:
            result = emit_tree(
                TaskTree(leaves=leaves),
                project=parsed.project,
                epic_slug=existing.epic_slug,
                runs_dir=settings.runs_dir,
            )
        except ArchitectError as e:
            print(f"Emission failed: {e}")
            return 1
        persist_tree(leaves, run_dir)

        print(f"\nTree emitted: {len(leaves)} leaf(s) for '{existing.epic_slug}'.")
        if result.created:
            print(f"  Created: {len(result.created)}")
        if result.skipped:
            print(f"  Skipped (existing): {len(result.skipped)}")
        print(f"Run ``meta build run {existing.epic_slug} --project {parsed.project}``.")
        return 0

    # --- plain path: A0+A1, then stop for the human -------------------------
    # Check for an existing framing BEFORE spending an architect call on a
    # proposal that persist_framing would refuse to write anyway.
    if framing_path.is_file() and not parsed.force:
        print(
            f"{framing_path} already exists — review/approve it, or pass --force "
            "to discard it and re-propose."
        )
        return 1

    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[A0] Inventorying project '{goal.project}'…")
    inv = collect_inventory(goal, repo=goal.repo or Path.cwd())
    if inv.truncated:
        print(f"  (truncated {inv.truncated} items to fit budget)")
    write_inventory(inv, run_dir)

    print("[A1] Proposing framing…")
    try:
        proposal = propose_framing(goal, inv)
    except ArchitectError as e:
        print(f"Framing failed: {e}")
        return 1
    try:
        persist_framing(proposal, run_dir, force=parsed.force)
    except ArchitectError as e:
        print(f"Framing write failed: {e}")
        return 1
    print(
        f"\nFraming proposal written to {run_dir}.\n"
        "Read framing.md; a human must set ``approved: true`` in framing.json before "
        "decomposition proceeds.\n"
        "Then run ``meta build plan <same spec> --approve`` to decompose and emit."
    )
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _cmd_run(argv: list[str]) -> int:
    """Orchestrator wave loop."""
    parser = argparse.ArgumentParser(
        prog="meta build run",
        description="Run the orchestrator wave loop for an epic.",
    )
    parser.add_argument(
        "epic_slug",
        help="The epic slug (used in pipeline-runs/<epic> and the integration branch).",
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Forge project name (framing does not carry it; there is no sane default).",
    )
    parser.add_argument(
        "--feature",
        default=None,
        help=(
            "Narrow the wave loop to one Feature value (default: the whole epic — "
            "every leaf carrying the pipeline:<epic>: ref)."
        ),
    )
    parser.add_argument(
        "--max-waves",
        type=int,
        default=None,
        help="Maximum number of waves to run (default: from settings).",
    )
    parser.add_argument(
        "--wave-gate",
        action="store_true",
        help="Stop after each wave for human review.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan waves and print, no dispatch/writes.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help=(
            "Leaves in flight per wave (default: settings.dispatch_concurrency). "
            "1 = serial; above 1 uses per-leaf jj workspaces + the reconcile barrier."
        ),
    )
    args = parser.parse_args(argv)

    run_dir = settings.runs_dir / args.epic_slug
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}. Run ``meta build plan`` first.")
        return 1

    require_approved_framing(run_dir)  # fail fast with the gate's own message

    result = run_epic(
        project=args.project,
        epic_slug=args.epic_slug,
        repo=Path.cwd(),
        feature=args.feature,
        max_waves=args.max_waves,
        wave_gate=args.wave_gate,
        dry_run=args.dry_run,
        concurrency=args.concurrency,
    )

    _print_orchestrator_result(result)
    return 0 if result.status != "aborted" else 1


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


def _cmd_gate(argv: list[str]) -> int:
    """The final epic gate: full-quorum sign-off on the whole epic diff. Never merges."""
    parser = argparse.ArgumentParser(
        prog="meta build gate",
        description="Run the full-quorum epic sign-off; approval means ready for HUMAN merge.",
    )
    parser.add_argument("epic_slug", help="Epic slug.")
    parser.add_argument("--main", default="main", help="The mainline branch/bookmark name.")
    args = parser.parse_args(argv)

    from forge.coding_pipeline.journal import append_gate_result
    from forge.coding_pipeline.vcs_epic import render_epic_gate, run_epic_gate

    run_dir = settings.runs_dir / args.epic_slug
    framing = require_approved_framing(run_dir)

    result = run_epic_gate(Path.cwd(), args.epic_slug, framing, main=args.main)
    append_gate_result(
        run_dir,
        "epic-signoff",
        result.approved,
        details=result.reason or ", ".join(result.providers),
    )
    print(render_epic_gate(result, args.epic_slug))
    return 0 if result.approved else 1


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def _cmd_status(argv: list[str]) -> int:
    """Tree summary from Forge + journal summary."""
    parser = argparse.ArgumentParser(
        prog="meta build status",
        description="Show pipeline status: tree summary, journal, and wave results.",
    )
    parser.add_argument(
        "epic_slug",
        nargs="?",
        default=None,
        help="The epic slug (default: last run).",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Forge project name (used to scope the Forge query; falls back to inventory.json).",
    )
    args = parser.parse_args(argv)

    run_dir = settings.runs_dir
    epic = args.epic_slug or _latest_epic(run_dir)
    if not epic:
        print("No epic slug provided and no prior runs found.")
        return 0

    # Resolve project: explicit flag > inventory.json fallback
    epic_dir = run_dir / epic
    if not args.project:
        inv_path = epic_dir / "inventory.json"
        if inv_path.is_file():
            try:
                inv_data = json.loads(inv_path.read_text())
                args.project = inv_data.get("project")
            except Exception:
                pass

    if not args.project:
        print("No --project specified and no inventory.json found.")
        return 1

    print(f"=== Pipeline status: {epic} (project={args.project}) ===\n")

    # Forge task tree — scope to project + include Done, then filter by epic ref.
    try:
        from nous_mcp.workflow import _query_tasks

        from forge.task_worker.nous_client import _read_db_content

        rows = _query_tasks(_read_db_content(), project=args.project, include_done=True, limit=None)
        epic_prefix = f"pipeline:{epic}:"
        features: dict[str, list[dict]] = {}
        for r in rows:
            ref = str(r.get("external_ref", "") or "")
            if not ref.startswith(epic_prefix):
                continue
            feat = str(r.get("feature", "") or "").strip()
            if not feat:
                continue
            features.setdefault(feat, []).append(r)
        for feat_name, tasks in features.items():
            print(f"  Feature: {feat_name}")
            for t in tasks:
                status = str(t.get("status", "?"))
                pri = t.get("priority", "?")
                ref = t.get("external_ref", "")
                print(f"    [{status:12s}] {t.get('task', '?'):50s} pri={pri} ref={ref}")
    except Exception:
        print("  (could not query Forge — Nous daemon may be unavailable)\n")

    # --- Journal summary --------------------------------------------------
    epics_dir = run_dir / epic
    if not epics_dir.is_dir():
        print("\nNo journal data found for this epic.")
        return 0

    # Waves
    waves = sorted(epics_dir.glob("wave-*.json"), key=lambda p: _wave_number(p.name))
    if waves:
        print(f"\n  Waves completed: {len(waves)}")
        for w in waves:
            try:
                record = WaveRecord.model_validate_json(w.read_text())
                landed = len(record.report.landed)
                failed = len(record.report.failed)
                print(
                    f"    wave-{record.wave:04d}: {landed} landed, {failed} failed, "
                    f"{len(record.report.findings)} findings"
                )
            except Exception:
                print(f"    wave-{_wave_number(w.name)}: (parse error)")
    else:
        print("\n  No waves run yet.")

    # Framing
    framing_path = epics_dir / "framing.json"
    if framing_path.is_file():
        try:
            fp = FramingProposal.model_validate_json(framing_path.read_text())
            approval = "approved" if fp.approved else "pending approval"
            print(f"\n  Framing: {approval}")
        except Exception:
            pass

    # Journal decision log
    journal_path = epics_dir / "journal.jsonl"
    if journal_path.is_file():
        lines = journal_path.read_text().strip().splitlines()
        if lines:
            print(f"\n  Journal entries: {len(lines)}")

    print()
    return 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _print_orchestrator_result(result: OrchestratorResult) -> None:
    """Pretty-print an orchestrator run result."""
    print(f"\nStatus: {result.status}")
    print(f"  Epic: {result.epic_slug}")
    print(f"  Waves run: {result.waves_run}")
    for note in result.notes:
        print(f"  - {note}")


def _wave_number(filename: str) -> int:
    m = re.match(r"wave-(\d+)\.json$", filename)
    return int(m.group(1)) if m else 0


def _latest_epic(runs_dir: Path) -> str | None:
    if not runs_dir.is_dir():
        return None
    entries = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for e in entries:
        if e.is_dir() and (e / "framing.json").is_file():
            return e.name
    return None


_COMMANDS = {
    "plan": _cmd_plan,
    "run": _cmd_run,
    "gate": _cmd_gate,
    "status": _cmd_status,
}


def main(argv: list[str] | None = None) -> int:
    """Dispatch ``meta build`` subcommands.

    Route-only: known subcommands go straight to their own parser with the remaining
    args (no double parse, no argument drift between an outer and inner parser, and
    ``argv=None`` correctly falls back to ``sys.argv``). The outer parser exists only
    to render help and reject unknown commands.
    """
    args = list(sys.argv[1:]) if argv is None else list(argv)
    if args and args[0] in _COMMANDS:
        return _COMMANDS[args[0]](args[1:])

    parser = argparse.ArgumentParser(
        prog="meta build",
        description="Coding pipeline: plan, run, and inspect epic builds.",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("plan", help="A0+A1 framing; --approve unlocks A2+A3.")
    sub.add_parser("run", help="Orchestrator wave loop.")
    sub.add_parser("gate", help="Final full-quorum epic sign-off (human merges after).")
    sub.add_parser("status", help="Tree + journal summary for an epic.")
    parser.parse_args(args)  # --help exits 0 here; unknown commands exit 2
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
