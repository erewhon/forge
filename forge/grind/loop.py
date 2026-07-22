"""The grind loop: iterate a goal via the runbook, checkpointing on the jj op log.

One turn = snapshot → let the model make one edit → run the cycle → score it → keep or roll back →
guard against no-progress → repeat. Bounded by ``max_iterations`` (the model spend lives in the
out-of-process OpenCode call, so iterations — not tokens — are the honest bound). Never commits.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from forge.coding_pipeline.journal import failure_signature
from forge.grind.executor import run_opencode_edit
from forge.grind.jj import current_op, ensure_jj, restore_op
from forge.grind.models import CycleResult, GrindConfig, IterationRecord
from forge.grind.prompt import build_spec
from forge.grind.runbook import run_cycle, score_improves
from forge.shared.lessons import draft_lesson, propose_lesson
from forge.task_worker.vcs import get_changed_files

# Injectable seams so tests can drive the loop without OpenCode or a real experiment.
EditFn = Callable[[Path, str, str, int], tuple[bool, str, bool]]
CycleFn = Callable[[GrindConfig, Path], CycleResult]


class GrindOutcome(BaseModel):
    status: Literal["already-done", "done", "stuck", "exhausted", "blocked"]
    iterations: int
    best_score: float | None
    summary: str


def _append_journal(run_dir: Path, record: IterationRecord) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "journal.jsonl").open("a") as fh:
        fh.write(record.model_dump_json() + "\n")


def grind(
    cfg: GrindConfig,
    repo: Path,
    *,
    model: str,
    run_dir: Path,
    log: Callable[[str], None] = print,
    edit_fn: EditFn = run_opencode_edit,
    cycle_fn: CycleFn = run_cycle,
) -> GrindOutcome:
    """Run the grind loop until the goal is met, the loop gets stuck, or the cap is hit."""
    ensure_jj(repo)
    hill_climb = cfg.check.score_regex is not None
    goal_dir = cfg.check.score_goal

    log("baseline: running the experiment cycle before any edit…")
    baseline = cycle_fn(cfg, repo)
    if baseline.passed:
        log("baseline already passes the check — nothing to grind.")
        return GrindOutcome(
            status="already-done",
            iterations=0,
            best_score=baseline.score,
            summary="The goal's check already passes; no changes made.",
        )

    best_score = baseline.score
    best_op = current_op(repo)
    observation = baseline.observation
    recent_sigs: list[str] = []

    for i in range(1, cfg.max_iterations + 1):
        log(f"── turn {i}/{cfg.max_iterations} " + ("─" * 32))
        pre_op = current_op(repo)
        spec = build_spec(cfg, repo, observation, i)
        ok, tail, blocked = edit_fn(repo, spec, model, cfg.edit_timeout)

        if blocked:
            log(f"model refused (BLOCKED). Rolling back turn {i}.\n{tail}")
            restore_op(repo, pre_op)
            _append_journal(
                run_dir,
                IterationRecord(
                    iteration=i,
                    edited_files=[],
                    blocked=True,
                    passed=False,
                    score=None,
                    failure_sig="",
                    kept=False,
                    reason="model refused (BLOCKED)",
                ),
            )
            return GrindOutcome(
                status="blocked",
                iterations=i,
                best_score=best_score,
                summary=f"Model refused on turn {i}: {tail.strip()[:200]}",
            )
        if not ok:
            log("opencode turn exited non-zero (advisory) — scoring the working copy anyway.")

        cycle = cycle_fn(cfg, repo)
        edited = _safe_changed(repo)
        sig = failure_signature(cycle.reason)

        if cycle.passed:
            _append_journal(
                run_dir,
                IterationRecord(
                    iteration=i,
                    edited_files=edited,
                    blocked=False,
                    passed=True,
                    score=cycle.score,
                    failure_sig="",
                    kept=True,
                    reason="",
                ),
            )
            log(f"✓ check passes on turn {i}. Goal met — state kept, nothing committed.")
            return GrindOutcome(
                status="done",
                iterations=i,
                best_score=cycle.score,
                summary=f"Goal met on turn {i} after editing {len(edited)} file(s).",
            )

        kept = True
        if hill_climb:
            if score_improves(cycle.score, best_score, goal_dir):
                best_score = cycle.score
                best_op = current_op(repo)
                log(f"  score improved → {cycle.score} (kept as best)")
            else:
                restore_op(repo, best_op)
                kept = False
                log(f"  score {cycle.score} did not beat best {best_score} → rolled back")
        else:
            best_op = current_op(repo)  # linear keep-last: this turn is the new tip

        _append_journal(
            run_dir,
            IterationRecord(
                iteration=i,
                edited_files=edited,
                blocked=False,
                passed=False,
                score=cycle.score,
                failure_sig=sig,
                kept=kept,
                reason=cycle.reason,
            ),
        )

        recent_sigs.append(sig)
        if _no_progress(recent_sigs, cfg.no_progress_window):
            lesson = draft_lesson(cycle.reason, count=cfg.no_progress_window)
            propose_lesson(run_dir, lesson)
            log(
                f"no progress: {cfg.no_progress_window} turns failed identically. Stopping and "
                f"proposing a lesson (see {run_dir / 'lessons.proposed.md'})."
            )
            return GrindOutcome(
                status="stuck",
                iterations=i,
                best_score=best_score,
                summary=f"Stuck after {i} turns — same failure {cfg.no_progress_window}× running: "
                f"{cycle.reason[:160]}",
            )

        observation = cycle.observation

    restore_op(repo, best_op)
    log(f"iteration cap ({cfg.max_iterations}) reached without meeting the goal. Best state kept.")
    return GrindOutcome(
        status="exhausted",
        iterations=cfg.max_iterations,
        best_score=best_score,
        summary=f"Ran {cfg.max_iterations} turns without meeting the goal; best state kept.",
    )


def _no_progress(sigs: list[str], window: int) -> bool:
    """True when the last *window* turns all failed with the same non-empty signature."""
    if len(sigs) < window:
        return False
    tail = sigs[-window:]
    return bool(tail[-1]) and all(s == tail[-1] for s in tail)


def _safe_changed(repo: Path) -> list[str]:
    try:
        return get_changed_files(repo)
    except Exception:  # noqa: BLE001 — a diff read must never sink the loop
        return []
