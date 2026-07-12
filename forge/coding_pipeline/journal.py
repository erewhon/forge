"""Wave journal and resume reconciliation for the coding pipeline.

Run directory layout (per epic):
    pipeline-runs/<epic>/framing.md|json, inventory.md, tree.json,
    wave-NNN.json, journal.jsonl

Mirrors the research harness layout and ``automerge.log_decision``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from forge.coding_pipeline.models import LeafOutcome, WaveRecord
from forge.shared.automerge import log_decision

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


def next_wave_number(runs_dir: Path, epic_slug: str) -> int:
    """The number the next wave should use: highest persisted wave + 1 (the
    research-harness resume mechanic — numbering continues across runs)."""
    return _highest_wave_number(runs_dir, epic_slug) + 1


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


def append_leaf_context(
    run_dir: Path,
    leaf_title: str,
    *,
    deps_landed: list[str],
    chars: int,
    error: str = "",
) -> Path:
    """One-line audit record of the epic-context preamble injected at dispatch
    (the preamble itself is ephemeral — it lives only in the dispatch spec file)."""
    record: dict[str, Any] = {
        "event": "leaf_context",
        "leaf": leaf_title,
        "deps_landed": deps_landed,
        "chars": chars,
    }
    if error:
        record["error"] = error
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
    # Full dump on purpose: the journal is the crash-recovery record, so explicit
    # defaults must survive the round trip rather than being dropped as "unset".
    path.write_text(record.model_dump_json(indent=2))
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


def landed_titles(run_dir: Path) -> set[str]:
    """Titles of every leaf this epic's journal records as landed (dispatch status "done").

    The journal is the source of truth for "this work is on the epic branch" — Forge
    status is not: a replan or a human can re-arm a landed leaf's task to Ready, and the
    planner/replanner use this set to treat landed leaves as terminal anyway (deps-v2
    waves 10-11: a respec of a landed leaf re-dispatched finished work, and the worker's
    no-change diagnostic then escalated the completed leaf to a human).
    """
    journal = _journal_path(run_dir)
    if not journal.exists():
        return set()
    titles: set[str] = set()
    for line in journal.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") == "leaf_dispatch" and rec.get("status") == "done" and rec.get("leaf"):
            titles.add(rec["leaf"])
    return titles


# ---------------------------------------------------------------------------
# Reconciliation
# ---------------------------------------------------------------------------


def reconcile(
    epic_slug: str,
    *,
    in_progress: Callable[[str], list[str]] | None = None,
    update_status: Callable[..., None] | None = None,
) -> list[str]:
    """Crash recovery: flip the epic's orphaned In Progress tasks back to Ready.

    The orchestrator calls this at startup, BEFORE dispatching anything. Under the
    single-orchestrator invariant (one run per repo — the dispatch lockfile enforces
    it), any task still In Progress at that moment is an orphan from a crashed or
    killed run: nothing live can be mid-dispatch. Each is returned to Ready with a
    diagnostic note and the titles are reported.

    Deliberately NOT gated on the run directory existing — the run dir (journal,
    wave records) persists across crashes by design, so its presence says nothing
    about whether a run is live.

    **Never touches the working copy** — the orchestrator owns VCS.

    ``in_progress`` (ref-prefix -> titles) and ``update_status`` (task, status,
    notes=...) are injectable for tests; the defaults go through the configured task
    store (Forge today, GitHub issues under the work-deployable adapter).
    """
    from forge.coding_pipeline.waves import epic_ref_prefix
    from forge.shared.task_store import get_task_store

    if in_progress is None:
        in_progress = get_task_store().in_progress_titles
    if update_status is None:
        update_status = get_task_store().update_status

    orphaned: list[str] = []
    for title in in_progress(epic_ref_prefix(epic_slug)):
        update_status(
            title,
            "Ready",
            notes=(
                "Reconciled by the coding pipeline at startup: task was In Progress "
                "with no live orchestrator run (crash recovery). Returned to Ready "
                "for redispatch."
            ),
        )
        orphaned.append(title)
    return orphaned
