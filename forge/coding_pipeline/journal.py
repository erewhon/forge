"""Wave journal and resume reconciliation for the coding pipeline.

Run directory layout (per epic):
    pipeline-runs/<epic>/framing.md|json, inventory.md, tree.json,
    wave-NNN.json, journal.jsonl

Mirrors the research harness layout and ``automerge.log_decision``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from agents.coding_pipeline.models import LeafOutcome, WaveRecord
from agents.shared.automerge import log_decision

_WAVE_RE = re.compile(r"^wave-(\d+)\.json$")


# ---------------------------------------------------------------------------
# Run directory helpers
# ---------------------------------------------------------------------------


def _wave_number(filename: str) -> int:
    """Extract the wave number from ``wave-NNN.json``."""
    m = _WAVE_RE.match(filename)
    if m is None:
        return -1
    return int(m.group(1))


def _highest_wave_number(runs_dir: Path, epic_slug: str) -> int:
    """Return the highest wave number currently persisted for *epic_slug*, or 0."""
    run_dir = runs_dir / epic_slug
    if not run_dir.is_dir():
        return 0
    max_n = 0
    for child in run_dir.iterdir():
        if child.is_file():
            n = _wave_number(child.name)
            if n > max_n:
                max_n = n
    return max_n


def _run_dir(runs_dir: Path, epic_slug: str) -> Path:
    """Return the run directory path, creating it if needed."""
    d = runs_dir / epic_slug
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Journal appender
# ---------------------------------------------------------------------------


def _journal_path(run_dir: Path) -> Path:
    return run_dir / "journal.jsonl"


def append_leaf_outcome(
    run_dir: Path,
    leaf_title: str,
    outcome: LeafOutcome,
) -> Path:
    """Append a single leaf outcome record to the decision log.

    The record is a JSONL line keyed by ``event: leaf_dispatch``.
    """
    record: dict[str, Any] = {
        "event": "leaf_dispatch",
        "leaf": leaf_title,
        "status": outcome.status,
    }
    if outcome.reason:
        record["reason"] = outcome.reason
    if outcome.commit_id:
        record["commit_id"] = outcome.commit_id
    log_decision(record, _journal_path(run_dir))
    return _journal_path(run_dir)


def append_gate_result(
    run_dir: Path,
    gate_name: str,
    passed: bool,
    details: str = "",
) -> Path:
    """Log a wave gate result (suite or review)."""
    record: dict[str, Any] = {
        "event": "gate_result",
        "gate": gate_name,
        "passed": passed,
    }
    if details:
        record["details"] = details
    log_decision(record, _journal_path(run_dir))
    return _journal_path(run_dir)


def append_replan_action(
    run_dir: Path,
    action_kind: str,
    **fields: Any,
) -> Path:
    """Log a replan action taken after a wave."""
    record: dict[str, Any] = {
        "event": "replan",
        "action": action_kind,
        **fields,
    }
    log_decision(record, _journal_path(run_dir))
    return _journal_path(run_dir)


def append_escalation(
    run_dir: Path,
    leaf_title: str,
    reason: str = "",
) -> Path:
    """Log an escalation (leaf hit the attempt cap)."""
    record: dict[str, Any] = {
        "event": "escalation",
        "leaf": leaf_title,
    }
    if reason:
        record["reason"] = reason
    log_decision(record, _journal_path(run_dir))
    return _journal_path(run_dir)


# ---------------------------------------------------------------------------
# WaveRecord persistence
# ---------------------------------------------------------------------------


def persist_wave(
    runs_dir: Path,
    epic_slug: str,
    record: WaveRecord,
) -> Path:
    """Serialize *record* to ``pipeline-runs/<epic>/wave-NNN.json``.

    Wave numbering continues from the highest existing wave file
    (research-harness resume mechanic).
    """
    run_dir = _run_dir(runs_dir, epic_slug)
    wave_n = record.wave
    path = run_dir / f"wave-{wave_n:04d}.json"
    path.write_text(record.model_dump_json(indent=2, exclude_unset=True))
    return path


def load_wave(
    runs_dir: Path,
    epic_slug: str,
    wave_n: int,
) -> WaveRecord | None:
    """Load a previously persisted wave record, or ``None`` if absent."""
    path = runs_dir / epic_slug / f"wave-{wave_n:04d}.json"
    if not path.exists():
        return None
    return WaveRecord.model_validate_json(path.read_text())


# ---------------------------------------------------------------------------
# Attempt counting from journal
# ---------------------------------------------------------------------------


def count_attempts(
    run_dir: Path,
    leaf_title: str,
) -> int:
    """Derive the number of times *leaf_title* has been dispatched by scanning
    the journal.  Returns ``0`` when the leaf has never been attempted.

    This means no separate mutable state file is needed — the journal IS the
    source of truth.
    """
    journal = _journal_path(run_dir)
    if not journal.exists():
        return 0
    count = 0
    for line in journal.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") == "leaf_dispatch" and rec.get("leaf") == leaf_title:
            count += 1
    return count


def count_attempts_for_all(
    run_dir: Path,
    leaf_titles: list[str],
) -> dict[str, int]:
    """Return a dict mapping every *leaf_title* in the list to its attempt count."""
    return {title: count_attempts(run_dir, title) for title in leaf_titles}


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def reconcile(
    runs_dir: Path,
    epic_slug: str,
    *,
    query_tasks: Any | None = None,
    update_task_status: Any | None = None,
    run_dir_exists: Any | None = None,
) -> list[str]:
    """Reconcile Forge task state with the persisted run directory.

    On startup the orchestrator calls this to find **orphaned** tasks — Forge
    tasks that are ``In Progress`` for this epic's feature but have no live run
    directory (crash recovery).  Those tasks are flipped back to ``Ready`` with
    a diagnostic note and the task names are returned.

    **Never touches the working copy** — the orchestrator owns VCS.

    Parameters
    ----------
    runs_dir:
        Top-level runs directory (the ``runs_dir`` config value).
    epic_slug:
        The epic whose tasks to reconcile.
    query_tasks:
        A callable matching ``query_tasks(status=…, feature=…)`` signature.
        Called as ``query_tasks(status="In Progress", feature=epic_slug)``.
        Defaults to importing the real ``nous_client.query_tasks``.
    update_task_status:
        A callable matching ``update_task_status(task, status, notes)``.
        Defaults to importing the real ``nous_client.update_task_status``.
    run_dir_exists:
        Override for checking whether the run dir exists (for testing).
        A callable ``(runs_dir, epic_slug) -> bool``.

    Returns
    -------
    list[str]
        Task names that were flipped back to Ready (empty list when nothing
        was orphaned).
    """
    run_dir = runs_dir / epic_slug
    if run_dir_exists is not None:
        exists = run_dir_exists(runs_dir, epic_slug)
    else:
        exists = run_dir.is_dir()

    # If a live run dir exists, nothing to reconcile — the orchestrator is
    # already tracking state.
    if exists:
        return []

    # Import inside the function so tests can mock the functions before calling.
    from agents.task_worker import nous_client

    if query_tasks is None:
        query_tasks = nous_client.query_tasks  # type: ignore[assignment]

    if update_task_status is None:
        update_task_status = nous_client.update_task_status  # type: ignore[assignment]

    # Find In Progress tasks for this feature/epic with no live run.
    # query_tasks returns compact rows; we look for status = "In Progress".
    in_progress = query_tasks(status="In Progress", feature=epic_slug)  # type: ignore[arg-type]

    orphaned: list[str] = []
    for row in in_progress:
        task_name = row.get("task", "")
        if not task_name:
            continue
        # Flip back to Ready with a diagnostic note.
        update_task_status(  # type: ignore[call-arg]
            task_name,
            "Ready",
            notes=(
                f"Reconciled by coding pipeline on crash recovery.\n"
                f"No live run directory found for epic `{epic_slug}`.\n"
                f"Task returned to Ready for redispatch."
            ),
        )
        orphaned.append(task_name)

    return orphaned
