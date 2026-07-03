"""Wave planning — the ready-set over Forge (design: "The wave loop").

Pure read-side logic: no LLM calls, no writes. ``plan_wave`` computes what the orchestrator
may dispatch next for an epic's feature — worker-ready leaves (Ready AND Auto-OK/Auto-Preferred
AND unblocked), Auto-Preferred first then priority ascending then title (the worker's own
selection order), capped at ``wave_size`` — plus the counts that let the loop distinguish
**dry** (tree exhausted → epic gate) from **waiting on humans** (Manual/Spec-Needed/blocked
leaves remain → report and exit cleanly).
"""

from __future__ import annotations

from agents.coding_pipeline.models import BlockedLeaf, LeafRow, WavePlan

_AUTO_MODES = {"auto-ok", "auto-preferred"}


def _mode_rank(mode: str) -> int:
    """Auto-Preferred before Auto-OK — mirrors the worker's own pick order."""
    return 0 if mode.strip().lower() == "auto-preferred" else 1


def _row_from_raw(raw: dict, blocked: bool, blocking: list[str]) -> LeafRow:
    """Normalize a ``_query_tasks`` row: null-as-manual, missing priority sorts last."""
    try:
        priority = int(raw.get("priority", 99))
    except (TypeError, ValueError):
        priority = 99
    return LeafRow(
        task=str(raw.get("task", "")),
        status=str(raw.get("status", "")),
        execution_mode=str(raw.get("execution_mode") or "Manual"),
        priority=priority,
        blocked=blocked,
        blocked_by=list(blocking),
    )


def fetch_feature_rows(project: str, feature: str) -> list[LeafRow]:
    """All task rows (any status, Done included) for *feature*, via the worker's
    daemon-backed Nous read path — no new HTTP code."""
    from nous_mcp.workflow import _is_task_blocked, _query_tasks

    from agents.task_worker.nous_client import _read_db_content

    db_content = _read_db_content()
    raw_rows = _query_tasks(
        db_content, project=project, feature=feature, include_done=True, limit=None
    )
    rows: list[LeafRow] = []
    for raw in raw_rows:
        blocked, blocking = _is_task_blocked(raw, db_content)
        row = _row_from_raw(raw, blocked, blocking)
        if row.task:
            rows.append(row)
    return rows


def plan_wave(
    feature: str,
    project: str,
    *,
    wave_size: int,
    rows: list[LeafRow] | None = None,
) -> WavePlan:
    """Compute the next wave's dispatch set and the epic's outstanding-work counts.

    ``rows`` is injectable for tests; the default reads Forge live. Worker-ready
    semantics match the worker exactly: status=Ready AND execution_mode in
    (Auto-OK, Auto-Preferred) AND unblocked.
    """
    if rows is None:
        rows = fetch_feature_rows(project, feature)

    dispatchable: list[LeafRow] = []
    ready_manual = 0
    spec_needed = 0
    in_progress = 0
    done = 0
    blocked: list[BlockedLeaf] = []

    for row in rows:
        status = row.status.strip().lower()
        mode = row.execution_mode.strip().lower()
        if status == "done":
            done += 1
            continue
        if row.blocked:
            blocked.append(BlockedLeaf(task=row.task, blocked_by=row.blocked_by))
            continue
        if status == "in progress":
            in_progress += 1
        elif status == "spec needed":
            spec_needed += 1
        elif status == "ready" and mode in _AUTO_MODES:
            dispatchable.append(row)
        elif status == "ready":
            ready_manual += 1

    dispatchable.sort(key=lambda r: (_mode_rank(r.execution_mode), r.priority, r.task))

    return WavePlan(
        feature=feature,
        project=project,
        dispatch=[row.task for row in dispatchable[:wave_size]],
        ready_manual=ready_manual,
        spec_needed=spec_needed,
        in_progress=in_progress,
        done=done,
        blocked=blocked,
    )
