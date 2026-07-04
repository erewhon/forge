"""CLI front door for the coding pipeline: ``meta build plan | run | status``.

Each subcommand is a thin wrapper around the pipeline's architectural layers.  ``main``
uses argparse subparsers so the CLI can be composed without Typer — each subcommand
is effectively its own entry point.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from agents.coding_pipeline.architect import (
    ArchitectError,
    decompose,
    persist_framing,
    persist_tree,
    propose_framing,
    require_approved_framing,
)
from agents.coding_pipeline.config import settings
from agents.coding_pipeline.emit import emit_tree
from agents.coding_pipeline.inventory import collect_inventory, write_inventory
from agents.coding_pipeline.models import FramingProposal, GoalSpec, TaskTree
from agents.coding_pipeline.orchestrator import OrchestratorResult, run_epic

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
    spec_path: Path | None = None
    if parsed.spec:
        spec_path = Path(parsed.spec)
        if not spec_path.is_file():
            print(f"Spec file not found: {spec_path}")
            return 1
        goal = GoalSpec.load(spec_path)
    else:
        goal = GoalSpec(goal="", project=parsed.project)

    # Derive run directory (shared with the orchestrator).
    epic = goal.epic_slug or "default"
    run_dir = settings.runs_dir / epic
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- A0: inventory ----------------------------------------------------
    print(f"[A0] Inventorying project '{goal.project}'…")
    try:
        repo_path = goal.repo if goal.repo else Path.cwd()
        inv = collect_inventory(goal, repo=repo_path)
    except ArchitectError as e:
        print(f"Inventory failed: {e}")
        return 1
    if inv.truncated:
        print(f"  (truncated {inv.truncated} items to fit budget)")

    # Persist inventory to the run dir.
    try:
        write_inventory(inv, run_dir)
    except Exception as e:
        print(f"Inventory write failed: {e}")

    # --- A1: framing proposal ---------------------------------------------
    print("[A1] Proposing framing…")
    try:
        proposal = propose_framing(goal, inv)
    except ArchitectError as e:
        print(f"Framing failed: {e}")
        return 1

    # --- --approve vs plain path ------------------------------------------
    if not parsed.approve:
        # Plain path: write framing proposal and exit.
        try:
            persist_framing(proposal, run_dir, force=parsed.force)
        except ArchitectError as e:
            print(f"Framing write failed: {e}")
            return 1
        print(
            "\nFraming proposal written to the run directory.\n"
            "A human must approve it (set ``approved: true`` in framing.json) before "
            "decomposition proceeds.\n"
            "Run ``meta build plan --approve`` once approved to continue."
        )
        return 0

    # --- --approve path: require existing approval -------------------------
    framing_path = run_dir / "framing.json"
    if not framing_path.is_file():
        print(
            "--approve was passed but no framing.json exists. "
            "Run without --approve first to produce the proposal."
        )
        return 1

    # Validate the existing framing is approved (never let --approve shortcut the gate).
    existing = FramingProposal.model_validate_json(framing_path.read_text())
    if not existing.approved:
        print(
            "--approve requires an already-approved framing.json. "
            "Either edit framing.json to set ``approved: true`` or approve interactively."
        )
        return 1

    # --- A2+A3: decomposition + emission ----------------------------------
    print("[A2] Decomposing tree…")
    try:
        leaves = decompose(existing, inv)
    except ArchitectError as e:
        print(f"Decomposition failed: {e}")
        return 1

    tree = TaskTree(leaves=leaves)

    print("[A3] Emitting task tree to Forge…")
    try:
        result = emit_tree(
            tree,
            project=parsed.project,
            epic_slug=goal.epic_slug or "default",
            runs_dir=settings.runs_dir,
        )
    except ArchitectError as e:
        print(f"Emission failed: {e}")
        return 1

    # Persist the tree for the orchestrator.
    try:
        persist_tree(leaves, run_dir)
    except Exception as e:
        print(f"Tree persistence failed: {e}")

    print(f"\nTree emitted: {len(leaves)} leaf(s) for '{goal.epic_slug}'.")
    if result.created:
        print(f"  Created: {len(result.created)}")
    if result.skipped:
        print(f"  Skipped (existing): {len(result.skipped)}")
    print("Run ``meta build run`` to start the orchestrator wave loop.")
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
        default=None,
        help="Forge project name (overrides the one in framing.json).",
    )
    parser.add_argument(
        "--feature",
        default=None,
        help="Feature name to scope leaves to (overrides the one in framing.json).",
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
    args = parser.parse_args(argv)

    run_dir = settings.runs_dir / args.epic_slug
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}. Run ``meta build plan`` first.")
        return 1

    framing = require_approved_framing(run_dir)
    project = args.project or framing.epic_slug
    feature = args.feature or "Coding Pipeline"

    repo = Path.cwd()
    if not repo.is_dir():
        repo = Path.home()

    result = run_epic(
        project=project,
        feature=feature,
        epic_slug=args.epic_slug,
        repo=repo,
        max_waves=args.max_waves,
        wave_gate=args.wave_gate,
        dry_run=args.dry_run,
    )

    _print_orchestrator_result(result)
    return 0 if result.status != "aborted" else 1


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
    args = parser.parse_args(argv)

    run_dir = settings.runs_dir
    epic = args.epic_slug or _latest_epic(run_dir)
    if not epic:
        print("No epic slug provided and no prior runs found.")
        return 0

    print(f"=== Pipeline status: {epic} ===\n")

    # Forge task tree (query Nous directly — same pattern as inventory.fetch_project_tasks).
    try:
        from nous_mcp.workflow import _query_tasks

        from agents.task_worker.nous_client import _read_db_content

        rows = _query_tasks(_read_db_content(), limit=None)
        # Group by feature
        features: dict[str, list[dict]] = {}
        for r in rows:
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
                print(f"    [{status:12s}] {t.get('task', '?'):50s} "
                      f"pri={pri} ref={ref}")
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
                print(f"    wave-{record.wave:04d}: {landed} landed, {failed} failed, "
                      f"{len(record.report.findings)} findings")
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


# Import WaveRecord at top level now that we're in helpers
from agents.coding_pipeline.models import WaveRecord  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Dispatch ``meta build`` subcommands."""
    parser = argparse.ArgumentParser(
        prog="meta build",
        description="Coding pipeline: plan, run, and inspect epic builds.",
    )
    sub = parser.add_subparsers(dest="command")

    # plan
    plan_p = sub.add_parser("plan", help="A0+A1 framing; --approve unlocks A2+A3.")
    plan_p.add_argument("spec", nargs="?", default=None, help="Goal spec file.")
    plan_p.add_argument("--project", required=True, help="Forge project name.")
    plan_p.add_argument("--approve", action="store_true", help="Approve framing + decompose.")
    plan_p.add_argument("--force", action="store_true", help="Overwrite existing approved framing.")

    # run
    run_p = sub.add_parser("run", help="Orchestrator wave loop.")
    run_p.add_argument("epic_slug", help="Epic slug.")
    run_p.add_argument("--project", default=None, help="Project override.")
    run_p.add_argument("--feature", default=None, help="Feature override.")
    run_p.add_argument("--max-waves", type=int, default=None, help="Max waves.")
    run_p.add_argument("--wave-gate", action="store_true", help="Stop per wave for review.")
    run_p.add_argument("--dry-run", action="store_true", help="Plan only, no dispatch.")

    # status
    sub.add_parser("status", help="Tree + journal summary for an epic.")

    args = parser.parse_args(argv)

    if args.command == "plan":
        argv = argv or []
        return _cmd_plan(argv[1:])  # strip "plan" verb, pass remaining to sub-parser
    if args.command == "run":
        argv = argv or []
        return _cmd_run(argv[1:])
    if args.command == "status":
        argv = argv or []
        return _cmd_status(argv[1:])

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
