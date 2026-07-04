"""The orchestrator wave loop — where the pipeline becomes one runnable thing.

Composes the built pieces per the design's "The wave loop":

    resume: reconcile orphaned In Progress tasks (crash recovery)
    while waves_this_run < max_waves:
        plan  = plan_wave(...)            # ready-set over Forge
        dry?  -> stop: the tree is exhausted, the epic gate is next
        waiting on humans? -> report and exit cleanly (never spin)
        outcomes = run_wave(plan)         # serial dispatch, repo lock, journal-as-you-go
        report   = verify_wave(...)       # suite hard gate + confirmed review findings
        actions  = replan(...)            # deterministic escalation + model judgment
        apply actions (idempotent emission / status updates), persist the WaveRecord
        halt action? -> stop.  --wave-gate? -> stop for human review.

Boundaries owned elsewhere: per-leaf safety lives in ``task_worker.run_one``; the per-repo
dispatch lock lives in ``dispatch``; epic-bookmark lifecycle and the final full-quorum epic
gate belong to the epic-branch leaf — this loop only asserts a clean working copy at start
and never moves VCS state. All Forge writes go through ``update_task_status`` and the
idempotent emission path; there are no ad-hoc row edits.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from agents.coding_pipeline.architect import (
    ArchitectError,
    deterministic_escalations,
    replan,
    require_approved_framing,
)
from agents.coding_pipeline.architect import load_tree as _load_tree
from agents.coding_pipeline.config import settings
from agents.coding_pipeline.dispatch import DispatchError, run_wave
from agents.coding_pipeline.emit import emit_fixup
from agents.coding_pipeline.journal import (
    append_escalation,
    append_gate_result,
    append_replan_action,
    count_attempts_for_all,
    next_wave_number,
    persist_wave,
    reconcile,
)
from agents.coding_pipeline.models import (
    EscalateAction,
    FixupAction,
    HaltAction,
    IntegrationFixAction,
    ReplanAction,
    RespecAction,
    SplitSubtreeAction,
    WaveRecord,
    WaveReport,
)
from agents.coding_pipeline.vcs_epic import ensure_epic_bookmark, update_epic_bookmark
from agents.coding_pipeline.verify import verify_wave, wave_start_rev
from agents.coding_pipeline.waves import fetch_feature_rows, plan_wave
from agents.task_worker.nous_client import update_task_status
from agents.task_worker.vcs import VCSError, get_changed_files

ExitStatus = Literal[
    "dry", "waiting-on-human", "planned", "wave-gate", "max-waves", "halted", "aborted"
]


class OrchestratorResult(BaseModel):
    """How a run ended and what it did — the CLI renders this."""

    status: ExitStatus
    epic_slug: str
    feature: str
    waves_run: int = 0
    dispatched: list[str] = []
    notes: list[str] = []


def _apply_actions(
    actions: list[ReplanAction],
    *,
    project: str,
    epic_slug: str,
    run_dir: Path,
    log: Callable[[str], None],
) -> bool:
    """Apply replan actions through the idempotent/append-only paths. Returns True on halt.

    Escalation flips status to Spec Needed — that alone removes the leaf from worker
    eligibility (the gate requires Ready); the execution-mode flip is a human triage nicety
    the current status-update path doesn't carry.
    """
    halted = False
    for action in actions:
        if isinstance(action, FixupAction | IntegrationFixAction):
            outcome = emit_fixup(action.leaf, project=project, epic_slug=epic_slug)
            append_replan_action(
                run_dir, action.kind, leaf=action.leaf.title, ref=outcome.external_ref
            )
            log(f"replan {action.kind}: {action.leaf.title} -> {outcome.action}")
        elif isinstance(action, RespecAction):
            update_task_status(
                action.leaf_title,
                "Ready",
                notes=(
                    f"## Respec (pipeline replan)\n\n{action.rationale}\n\n"
                    f"Revised spec for the next attempt:\n\n{action.revised.content}"
                ),
            )
            append_replan_action(run_dir, "respec", leaf=action.leaf_title)
            log(f"replan respec: {action.leaf_title}")
        elif isinstance(action, EscalateAction):
            update_task_status(
                action.leaf_title,
                "Spec Needed",
                notes=(
                    "Escalated by the pipeline at the attempt cap — needs a human.\n\n"
                    f"Diagnostics:\n\n{action.diagnostics}"
                ),
            )
            append_escalation(run_dir, action.leaf_title, action.diagnostics)
            log(f"replan escalate: {action.leaf_title} -> Spec Needed")
        elif isinstance(action, SplitSubtreeAction):
            for row in fetch_feature_rows(project, action.feature):
                if row.status.strip().lower() == "ready":
                    update_task_status(
                        row.task,
                        "Spec Needed",
                        notes=(
                            f"Parked by the pipeline for subtree re-decomposition: "
                            f"{action.rationale}"
                        ),
                    )
            append_replan_action(
                run_dir, "split_subtree", feature=action.feature, rationale=action.rationale
            )
            log(f"replan split_subtree: {action.feature}")
        elif isinstance(action, HaltAction):
            append_replan_action(run_dir, "halt", reason=action.reason)
            log(f"replan HALT: {action.reason}")
            halted = True
    return halted


def run_epic(
    *,
    project: str,
    feature: str,
    epic_slug: str,
    repo: Path,
    max_waves: int | None = None,
    wave_gate: bool = False,
    dry_run: bool = False,
    log: Callable[[str], None] = print,
) -> OrchestratorResult:
    """Run up to ``max_waves`` waves of the epic; re-run to continue (numbering resumes).

    Requires a human-approved framing in the run dir (the A1 gate holds here too) and a
    clean working copy (positioning is the epic-branch leaf's job). ``dry_run`` plans the
    next wave and stops — no dispatch, no writes.
    """
    limit = max_waves if max_waves is not None else settings.max_waves
    run_dir = settings.runs_dir / epic_slug
    run_dir.mkdir(parents=True, exist_ok=True)

    framing = require_approved_framing(run_dir)
    tree = _load_tree(run_dir) or []
    result = OrchestratorResult(status="max-waves", epic_slug=epic_slug, feature=feature)

    orphaned = reconcile(feature)
    if orphaned:
        result.notes.append(f"reconciled orphaned In Progress tasks: {', '.join(orphaned)}")
        log(result.notes[-1])

    if not dry_run:
        try:
            ensure_epic_bookmark(repo, epic_slug, log=log)
        except VCSError as e:
            result.status = "aborted"
            result.notes.append(f"epic bookmark setup failed: {e}")
            log(result.notes[-1])
            return result

    while result.waves_run < limit:
        plan = plan_wave(feature, project, wave_size=settings.wave_size)
        if plan.dry:
            result.status = "dry"
            result.notes.append("tree exhausted — ready for the epic gate")
            log(result.notes[-1])
            return result
        if plan.waiting_on_human:
            result.status = "waiting-on-human"
            result.notes.append(
                f"nothing dispatchable: {plan.ready_manual} manual, {plan.spec_needed} "
                f"spec-needed, {len(plan.blocked)} blocked leaf(s) outstanding"
            )
            log(result.notes[-1])
            return result
        if dry_run:
            result.status = "planned"
            result.dispatched = plan.dispatch
            result.notes.append(f"[dry-run] would dispatch: {', '.join(plan.dispatch)}")
            log(result.notes[-1])
            return result

        wave_n = next_wave_number(settings.runs_dir, epic_slug)
        log(f"=== wave {wave_n}: dispatching {len(plan.dispatch)} leaf(s) ===")
        if get_changed_files(repo):
            result.status = "aborted"
            result.notes.append("working copy not clean at wave start — aborting untouched")
            log(result.notes[-1])
            return result
        wave_start = wave_start_rev(repo)

        try:
            outcomes = run_wave(plan, repo, journal_dir=run_dir, log=log)
        except DispatchError as e:
            append_gate_result(run_dir, "dispatch-preflight", False, details=str(e))
            result.status = "aborted"
            result.notes.append(f"wave aborted: {e}")
            log(result.notes[-1])
            return result

        report: WaveReport = verify_wave(repo, wave=wave_n, from_change=wave_start)
        report.outcomes = outcomes
        suite_tail = report.suite.output_tail if report.suite else ""
        append_gate_result(
            run_dir, "suite", report.suite_green, details="" if report.suite_green else suite_tail
        )
        confirmed = [f for f in report.findings if f.confirmed]
        append_gate_result(
            run_dir,
            "review",
            True,  # advisory: never blocks, findings feed replan
            details=f"{len(confirmed)} confirmed finding(s) of {len(report.findings)}",
        )

        attempts = count_attempts_for_all(run_dir, [o.leaf for o in report.failed])
        try:
            actions = replan(framing, tree, report, attempts)
        except ArchitectError as e:
            # A failed model replan must not kill the wave (e2e dry-run: it crashed the
            # run twice, losing the wave record while the journal kept counting
            # attempts). Degrade to the deterministic pre-rules — capped leaves still
            # escalate — and journal the degradation so a human sees replan is limping.
            actions = list(deterministic_escalations(report, attempts))
            append_replan_action(run_dir, "replan-degraded", reason=str(e))
            result.notes.append(f"replan degraded to deterministic escalations: {e}")
            log(result.notes[-1])
        halted = _apply_actions(
            actions, project=project, epic_slug=epic_slug, run_dir=run_dir, log=log
        )

        persist_wave(
            settings.runs_dir,
            epic_slug,
            WaveRecord(wave=wave_n, dispatched=plan.dispatch, report=report, actions=actions),
        )
        try:
            update_epic_bookmark(repo, epic_slug, log=log)  # the wave checkpoint re-push
        except VCSError as e:
            result.notes.append(f"warning: epic bookmark update failed: {e}")
            log(result.notes[-1])
        result.waves_run += 1
        result.dispatched.extend(plan.dispatch)

        if halted:
            result.status = "halted"
            result.notes.append("replan halted the run — the framing needs human review")
            log(result.notes[-1])
            return result
        if wave_gate:
            result.status = "wave-gate"
            result.notes.append("stopping at the wave gate for human review")
            log(result.notes[-1])
            return result

    result.notes.append(f"max waves for this run ({limit}) — re-run to continue")
    log(result.notes[-1])
    return result
