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

from forge.coding_pipeline.architect import (
    ArchitectError,
    deterministic_escalations,
    replan,
    require_approved_framing,
)
from forge.coding_pipeline.architect import load_tree as _load_tree
from forge.coding_pipeline.config import settings
from forge.coding_pipeline.context import build_leaf_context
from forge.coding_pipeline.dispatch import DispatchError, run_wave
from forge.coding_pipeline.emit import emit_fixup
from forge.coding_pipeline.journal import (
    append_budget_exhausted,
    append_escalation,
    append_gate_result,
    append_lesson_proposed,
    append_replan_action,
    count_attempts_for_all,
    landed_titles,
    next_wave_number,
    persist_wave,
    reconcile,
    recurring_failures,
    stuck_leaves,
)
from forge.coding_pipeline.journal_mirror import hydrate_run_dir, mirror_framing, mirror_run_dir
from forge.coding_pipeline.models import (
    EscalateAction,
    FixupAction,
    HaltAction,
    IntegrationFixAction,
    LeafOutcome,
    ReplanAction,
    RespecAction,
    SplitSubtreeAction,
    WaveRecord,
    WaveReport,
)
from forge.coding_pipeline.reconcile import apply_bisect
from forge.coding_pipeline.vcs_epic import ensure_epic_bookmark, update_epic_bookmark
from forge.coding_pipeline.verify import verify_wave, wave_start_rev
from forge.coding_pipeline.waves import fetch_epic_rows, fetch_feature_rows, plan_wave
from forge.shared.lessons import draft_lesson, propose_lesson
from forge.shared.task_store import TaskStore, get_task_store
from forge.shared.usage import UsageLedger, active_ledger
from forge.task_worker.tester import run_tests
from forge.task_worker.vcs import VCSError, get_changed_files

ExitStatus = Literal[
    "dry",
    "waiting-on-human",
    "planned",
    "wave-gate",
    "max-waves",
    "halted",
    "aborted",
    "budget-exhausted",
]


class OrchestratorResult(BaseModel):
    """How a run ended and what it did — the CLI renders this."""

    status: ExitStatus
    epic_slug: str
    feature: str = ""  # the optional narrowing filter; "" = the whole epic
    waves_run: int = 0
    dispatched: list[str] = []
    notes: list[str] = []
    total_tokens: int = 0  # the run's cumulative pipeline API spend (see UsageLedger)


def _propose_repo_lessons(run_dir: Path, *, log: Callable[[str], None]) -> None:
    """Hill-climbing hook: for each failure class the epic has hit twice, draft a one-line lesson
    and record it as a PROPOSAL (visible artifact + journal) — never a silent append to the active
    lessons file. Best-effort; a proposal failure must not disturb the wave."""
    try:
        for signature, count, reason in recurring_failures(run_dir):
            lesson = draft_lesson(reason, count=count)
            if propose_lesson(run_dir, lesson):  # new (deduped against earlier proposals)
                append_lesson_proposed(run_dir, lesson, signature=signature, count=count)
                log(f"proposed lesson (failure class seen {count}×): {lesson}")
    except Exception as exc:  # noqa: BLE001 — the rules layer never breaks the loop
        log(f"warning: lesson proposal failed: {exc}")


def _settle_landed_noops(
    outcomes: list[LeafOutcome],
    journal_landed: set[str],
    *,
    store: TaskStore,
    run_dir: Path,
    log: Callable[[str], None],
) -> None:
    """Convert a no-change failure on an already-landed leaf into completion.

    A leaf the journal records as landed whose re-dispatch produced no file changes
    is finished work wearing a failure label: its diff is already on the epic branch.
    Mark the Forge task Done and flip the outcome to skipped so replan neither burns
    an attempt nor escalates a completed leaf (deps-v2 wave 11, live). Mutates
    ``outcomes`` in place.
    """
    for outcome in outcomes:
        if (
            outcome.status == "failed"
            and outcome.leaf in journal_landed
            and "no file changes" in outcome.reason
        ):
            store.update_status(
                outcome.leaf,
                "Done",
                notes=(
                    "Marked Done by the orchestrator: this leaf already landed on the "
                    "epic branch (journal record) and its re-dispatch correctly produced "
                    "no changes. The failure label was the redispatch's, not the work's."
                ),
            )
            outcome.status = "skipped"
            outcome.reason = "already landed; no-change re-dispatch settled as Done"
            append_replan_action(run_dir, "landed-noop-settled", leaf=outcome.leaf)
            log(f"already landed — settled as Done: {outcome.leaf}")


def _apply_actions(
    actions: list[ReplanAction],
    *,
    project: str,
    epic_slug: str,
    run_dir: Path,
    store: TaskStore,
    log: Callable[[str], None],
    existing_titles: set[str] | frozenset[str] = frozenset(),
) -> bool:
    """Apply replan actions through the idempotent/append-only paths. Returns True on halt.

    Escalation flips status to Spec Needed AND execution mode to Manual: Spec Needed
    alone removes worker eligibility (the gate requires Ready), but without the mode
    flip a human re-arming the task to Ready would silently re-enter the auto pool —
    the design doc's "Spec Needed + Manual" is now what actually lands in Forge.

    ``existing_titles`` (lowercased, the epic's current task titles) is the exact-title
    dedup backstop under the ref-keyed dedup: a re-discovered finding filed under a new
    slug must not create a same-title twin — duplicate titles break every title-resolving
    task tool downstream (deps-v2 waves 17-26, live).
    """
    halted = False
    for action in actions:
        if isinstance(action, FixupAction | IntegrationFixAction):
            if action.leaf.title.strip().lower() in existing_titles:
                append_replan_action(
                    run_dir, "fixup-dedup-skip", leaf=action.leaf.title, reason="duplicate title"
                )
                log(f"replan {action.kind}: title already filed in this epic — skipped")
                continue
            # Fix-ups key their ref on the finding's slug (stable across replans);
            # integration fixes have no finding and fall back to the title slug.
            outcome = emit_fixup(
                action.leaf,
                project=project,
                epic_slug=epic_slug,
                finding_slug=getattr(action, "finding_slug", None),
            )
            append_replan_action(
                run_dir, action.kind, leaf=action.leaf.title, ref=outcome.external_ref
            )
            log(f"replan {action.kind}: {action.leaf.title} -> {outcome.action}")
        elif isinstance(action, RespecAction):
            store.update_status(
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
            store.update_status(
                action.leaf_title,
                "Spec Needed",
                notes=(
                    "Escalated by the pipeline (attempt cap or no-progress guard) — needs a "
                    f"human.\n\nDiagnostics:\n\n{action.diagnostics}"
                ),
                execution_mode="Manual",
            )
            append_escalation(run_dir, action.leaf_title, action.diagnostics)
            log(f"replan escalate: {action.leaf_title} -> Spec Needed")
        elif isinstance(action, SplitSubtreeAction):
            for row in fetch_feature_rows(project, action.feature):
                if row.status.strip().lower() == "ready":
                    store.update_status(
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
    epic_slug: str,
    repo: Path,
    feature: str | None = None,
    max_waves: int | None = None,
    wave_gate: bool = False,
    dry_run: bool = False,
    concurrency: int | None = None,
    log: Callable[[str], None] = print,
) -> OrchestratorResult:
    """Run the epic under a token ledger, then report cumulative spend.

    Thin wrapper: it binds an ambient :class:`UsageLedger` (persisted at ``<run_dir>/usage.json``,
    so an epic's spend accumulates across waves and resumes across invocations) for the duration of
    the run, so every in-process LLM call underneath counts, then stamps the total onto the result.
    The wave loop and its budget guard live in :func:`_run_epic`."""
    ledger = UsageLedger(settings.runs_dir / epic_slug / "usage.json")
    with active_ledger(ledger):
        result = _run_epic(
            project=project,
            epic_slug=epic_slug,
            repo=repo,
            feature=feature,
            max_waves=max_waves,
            wave_gate=wave_gate,
            dry_run=dry_run,
            concurrency=concurrency,
            ledger=ledger,
            log=log,
        )
    result.total_tokens = ledger.usage.total_tokens
    if ledger.usage.total_tokens and result.status != "budget-exhausted":
        result.notes.append(
            f"pipeline API spend: {ledger.usage.total_tokens} tokens across "
            f"{ledger.usage.calls} calls"
        )
    return result


def _run_epic(
    *,
    project: str,
    epic_slug: str,
    repo: Path,
    feature: str | None = None,
    max_waves: int | None = None,
    wave_gate: bool = False,
    dry_run: bool = False,
    concurrency: int | None = None,
    ledger: UsageLedger,
    log: Callable[[str], None] = print,
) -> OrchestratorResult:
    """Run up to ``max_waves`` waves of the epic; re-run to continue (numbering resumes).

    Scope is the whole epic (external_ref prefix ``pipeline:{epic_slug}:`` — every
    feature the decomposition or a replan produced); ``feature`` narrows to one
    Feature value when given. Requires a human-approved framing in the run dir (the
    A1 gate holds here too) and a clean working copy (positioning is the epic-branch
    leaf's job). ``dry_run`` plans the next wave and stops — no dispatch, no writes.
    """
    limit = max_waves if max_waves is not None else settings.max_waves
    run_dir = settings.runs_dir / epic_slug
    run_dir.mkdir(parents=True, exist_ok=True)
    store = get_task_store()

    # Resume-from-clone: if this machine has the epic's mirror ref but no local run dir, rebuild
    # the run dir from refs/pipeline/<epic> so the framing/journal/wave reads below find it. A
    # no-op when local state already exists (write-primary) or the ref is absent.
    hydrate_run_dir(repo, run_dir, epic_slug, log=log)

    framing = require_approved_framing(run_dir)
    tree = _load_tree(run_dir) or []
    result = OrchestratorResult(status="max-waves", epic_slug=epic_slug, feature=feature or "")

    orphaned = reconcile(epic_slug)
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
        # Record the approved framing as the FIRST commit of refs/pipeline/<epic>, before any
        # wave — so the repo carries proof that a human authorized this scope before work began.
        # Best-effort; a re-approved framing appends rather than rewrites.
        mirror_framing(repo, run_dir, epic_slug, log=log)

    while result.waves_run < limit:
        # Spend guard (checked at each wave boundary, before more work): an attempt cap bounds
        # retries, not cost. Fail closed — park with a journal entry, resumable (the ledger and
        # tasks persist; a re-run reloads the spent total and parks again until the budget lifts).
        budget = settings.epic_token_budget
        if budget is not None and ledger.usage.total_tokens >= budget:
            result.status = "budget-exhausted"
            used = ledger.usage.total_tokens
            append_budget_exhausted(run_dir, used=used, budget=budget, waves_run=result.waves_run)
            result.notes.append(
                f"token budget exhausted: {used}/{budget} tokens spent after "
                f"{result.waves_run} wave(s) — parked (resumable; raise the budget to continue)"
            )
            log(result.notes[-1])
            return result

        # One Forge read per wave, shared by the planner and the epic-context builder.
        epic_rows = fetch_epic_rows(project, epic_slug, feature=feature)
        # The journal's landed set is the terminal-work override for this wave: the
        # planner never redispatches it, and replan may not respec/escalate it.
        journal_landed = landed_titles(run_dir)
        plan = plan_wave(
            epic_slug,
            project,
            wave_size=settings.wave_size,
            feature=feature,
            rows=epic_rows,
            landed_titles=journal_landed,
        )
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

        def _preamble(task, _rows=epic_rows):
            return build_leaf_context(
                task,
                run_dir=run_dir,
                repo=repo,
                epic_goal=framing.restated_goal,
                siblings=_rows,
            )

        try:
            outcomes = run_wave(
                plan,
                repo,
                journal_dir=run_dir,
                preamble_for=_preamble,
                concurrency=concurrency,
                wave=wave_n,
                log=log,
            )
        except DispatchError as e:
            append_gate_result(run_dir, "dispatch-preflight", False, details=str(e))
            result.status = "aborted"
            result.notes.append(f"wave aborted: {e}")
            log(result.notes[-1])
            return result

        open_fixups = [
            row.task
            for row in epic_rows
            if f":{epic_slug}:fix:" in row.external_ref and row.status.strip().lower() != "done"
        ]
        report: WaveReport = verify_wave(
            repo, wave=wave_n, from_change=wave_start, existing_fixups=open_fixups
        )
        report.outcomes = outcomes
        # journal_landed was read BEFORE this wave dispatched, so it holds only prior
        # waves' landings — exactly the set a no-change "failure" should settle against.
        _settle_landed_noops(report.outcomes, journal_landed, store=store, run_dir=run_dir, log=log)
        suite_tail = report.suite.output_tail if report.suite else ""
        append_gate_result(
            run_dir, "suite", report.suite_green, details="" if report.suite_green else suite_tail
        )
        # A red CONCURRENT wave with several landed leaves is unattributed — serial dispatch
        # would have caught the breaking leaf at its own suite run. Bisect restores parity:
        # back out the first-red leaf, demote it, and let replan see a failure, not a mystery.
        effective_cap = concurrency if concurrency is not None else settings.dispatch_concurrency
        bisected = apply_bisect(
            report,
            repo=repo,
            base_rev=wave_start,
            concurrency=effective_cap,
            run_suite=lambda: run_tests(repo),
            on_demote=lambda title, note: store.update_status(title, "Ready", notes=note),
            log=log,
        )
        if bisected is not None:
            append_gate_result(
                run_dir,
                "bisect",
                report.suite_green,
                details=(
                    f"offender: {bisected.offender}; suite after back-out: "
                    f"{'green' if bisected.repaired_green else 'still red'}"
                ),
            )
        confirmed = [f for f in report.findings if f.confirmed]
        review_detail = (
            f"{len(confirmed)} confirmed of {len(report.findings)} canonical "
            f"(raw {report.raw_findings}, {len(report.dropped_covered)} covered by open fixups"
            + ("" if report.consolidation_ok else ", consolidation FAILED OPEN")
            + ")"
        )
        append_gate_result(
            run_dir,
            "review",
            True,  # advisory: never blocks, findings feed replan
            details=review_detail,
        )

        failed_titles = [o.leaf for o in report.failed]
        attempts = count_attempts_for_all(run_dir, failed_titles)
        # No-progress guard: leaves whose last two attempts failed identically escalate now,
        # before they burn the rest of their attempt budget repeating the same mistake.
        stuck = stuck_leaves(run_dir, failed_titles)
        try:
            actions = replan(
                framing, tree, report, attempts, landed_titles=journal_landed, stuck=stuck
            )
        except ArchitectError as e:
            # A failed model replan must not kill the wave (e2e dry-run: it crashed the
            # run twice, losing the wave record while the journal kept counting
            # attempts). Degrade to the deterministic pre-rules — capped/stuck leaves still
            # escalate — and journal the degradation so a human sees replan is limping.
            actions = list(deterministic_escalations(report, attempts, stuck))
            # Journal the model's raw output too — that's the only record of *why* replan's
            # judgment half never fires (see pipeline:build:fix:replan-validation). Keep it out
            # of the console note, which stays a one-liner.
            append_replan_action(
                run_dir, "replan-degraded", reason=str(e), raw_output=getattr(e, "raw", "")
            )
            result.notes.append(f"replan degraded to deterministic escalations: {e}")
            log(result.notes[-1])
        halted = _apply_actions(
            actions,
            project=project,
            epic_slug=epic_slug,
            run_dir=run_dir,
            store=store,
            log=log,
            existing_titles={row.task.strip().lower() for row in epic_rows},
        )

        # Hill-climbing (rules layer): a failure class the loop has now hit twice becomes a
        # PROPOSED lesson — a visible artifact a human promotes into the repo's lessons file.
        # Before persist_wave so the proposal rides into the wave's provenance snapshot.
        _propose_repo_lessons(run_dir, log=log)

        persist_wave(
            settings.runs_dir,
            epic_slug,
            WaveRecord(wave=wave_n, dispatched=plan.dispatch, report=report, actions=actions),
        )
        # Mirror the just-updated run dir into refs/pipeline/<epic> as one append-only commit —
        # the decision history now travels with the repo. Best-effort; the checkpoint push below
        # carries the new ref (push_pipeline_refs picks it up).
        mirror_run_dir(repo, run_dir, epic_slug, message=f"wave {wave_n}: {epic_slug}", log=log)
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
