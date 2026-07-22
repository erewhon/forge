"""The drain loop — repeatedly pick the top worker-ready leaf and run it through the existing task
worker, until the queue is dry (or a cap is hit).

Two subtleties this loop exists to handle correctly:

- **A failed leaf reverts to Ready.** So a naive ``next_ready`` loop would re-pick the same failing
  head forever and never reach lower leaves. We instead take the *full* dispatchable set each pass,
  exclude a ``seen`` set (a failed leaf lands in it and is skipped), and stop only when nothing new
  remains.
- **Draining a leaf can unblock its dependents.** So we re-query every pass rather than snapshotting
  once — a dependent that becomes Ready mid-window is picked up.

Readiness is not reimplemented here: :attr:`forge.queue.models.QueueRow.is_dispatchable` *is* the
task worker's gate (Ready ∧ auto ∧ unblocked), resolved from the same normalized rows.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from forge.queue.models import QueueRow
from forge.switcheroo.models import LeafOutcome

# Injected for testing; the defaults resolve the real (heavy) collaborators lazily so importing
# this module — and unit-testing with fakes — never pulls in the worker's OpenCode/sandbox stack.
StoreLike = object
RunOne = Callable[..., object]


def _rank(row: QueueRow) -> tuple[int, int]:
    """Sort key: Auto-Preferred ahead of Auto-OK, then by priority (lower first) — the worker's own
    ordering."""
    return (0 if row.execution_mode == "Auto-Preferred" else 1, row.priority)


def worker_ready_rows(store: StoreLike, allowed: list[str]) -> list[QueueRow]:
    """Every dispatchable leaf the worker would pick up, across projects, in worker order. *allowed*
    (empty = all) narrows to specific projects."""
    allowed_lower = {p.lower() for p in allowed}
    rows = [
        r
        for r in store.queue()
        if r.is_dispatchable and (not allowed_lower or r.project.lower() in allowed_lower)
    ]
    rows.sort(key=_rank)
    return rows


def _leaf_from_outcome(outcome: object, at: str) -> LeafOutcome:
    """Project a task worker ``RunOutcome`` into a journalled :class:`LeafOutcome`. Duck-typed so
    the heavy ``forge.task_worker`` import stays out of this module and its tests."""
    return LeafOutcome(
        task=outcome.task,
        project=outcome.project,
        status=outcome.status,
        reason=outcome.reason,
        commit_id=outcome.commit_id,
        changed_files=list(outcome.changed_files),
        duration_s=outcome.duration_s,
        at=at,
    )


def _default_run_one() -> RunOne:
    from forge.task_worker.main import run_one

    return run_one


def drain(
    *,
    store: StoreLike,
    allowed: list[str],
    max_leaves: int = 0,
    run_one_fn: RunOne | None = None,
    on_outcome: Callable[[LeafOutcome], None] | None = None,
) -> list[LeafOutcome]:
    """Drain worker-ready leaves through the task worker until the queue is dry.

    *max_leaves* caps the count (0 = unbounded — stop when nothing is ready). *run_one_fn* defaults
    to the real ``forge.task_worker.main.run_one``; tests inject a fake. *on_outcome* fires after
    each leaf (the CLI uses it to journal + print incrementally, so an interrupted window is still
    recorded). Each leaf resolves its own checkout inside ``run_one`` — the loop passes no ``repo``.
    """
    run_one_fn = run_one_fn or _default_run_one()
    seen: set[tuple[str, str]] = set()
    leaves: list[LeafOutcome] = []

    while max_leaves == 0 or len(leaves) < max_leaves:
        ready = [r for r in worker_ready_rows(store, allowed) if (r.project, r.task) not in seen]
        if not ready:
            break
        row = ready[0]
        seen.add((row.project, row.task))
        at = datetime.now(UTC).isoformat(timespec="seconds")

        task = store.find_task(row.task)
        if task is None or task.project != row.project:
            # A title that no longer resolves to this project's checkout (e.g. a cross-project title
            # collision). Record it as skipped rather than silently dropping it, and move on.
            leaf = LeafOutcome(
                task=row.task,
                project=row.project,
                status="skipped",
                reason="could not resolve leaf to a checkout",
                at=at,
            )
        else:
            leaf = _leaf_from_outcome(run_one_fn(task, repo=None), at)

        leaves.append(leaf)
        if on_outcome is not None:
            on_outcome(leaf)

    return leaves
