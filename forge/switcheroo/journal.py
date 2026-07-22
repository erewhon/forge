"""The failover journal: durable, incrementally-written record of a switcheroo window, living
beside the baton in the home repo's ``.forge/``. Each drained leaf is persisted as it completes, so
an interrupted window (the whole point of a *failover* tool) still leaves switch-back a true record.

One window is active at a time (``failover.json``); starting a new one archives the previous window
to ``history/`` rather than clobbering it, so the trail survives.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from forge.shared.baton import Baton
from forge.switcheroo.models import FailoverLog, LeafOutcome

#: Switcheroo's corner of the machine-managed ``.forge/`` dir (shared with baton.md, lessons.md).
SWITCHEROO_DIR = ".forge/switcheroo"
FAILOVER_NAME = "failover.json"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def switcheroo_dir(repo: Path) -> Path:
    return repo / SWITCHEROO_DIR


def failover_path(repo: Path) -> Path:
    return switcheroo_dir(repo) / FAILOVER_NAME


def history_dir(repo: Path) -> Path:
    return switcheroo_dir(repo) / "history"


def read_failover(repo: Path) -> FailoverLog | None:
    """The active failover window for *repo*, or ``None`` when there is none / it's unreadable."""
    path = failover_path(repo)
    if not path.is_file():
        return None
    try:
        return FailoverLog.model_validate_json(path.read_text())
    except ValueError:
        return None


def write_failover(repo: Path, log: FailoverLog) -> None:
    """Persist *log* as the active window (atomic-ish: full rewrite each call)."""
    path = failover_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(log.model_dump_json(indent=2))


def _archive(repo: Path, log: FailoverLog) -> None:
    """Move a prior window into ``history/<started_at>.json`` so a new window never clobbers it."""
    stamp = re.sub(r"[^0-9A-Za-z-]", "-", log.started_at) or _now()
    hist = history_dir(repo)
    hist.mkdir(parents=True, exist_ok=True)
    (hist / f"{stamp}.json").write_text(log.model_dump_json(indent=2))


def start_failover(repo: Path, *, baton: Baton, model_tier: str, reason: str = "") -> FailoverLog:
    """Open a fresh failover window anchored to *baton*, archiving any prior active window."""
    prior = read_failover(repo)
    if prior is not None:
        _archive(repo, prior)
    log = FailoverLog(
        started_at=_now(),
        reason=reason,
        model_tier=model_tier,
        baton_goal=baton.goal,
        baton_vcs=baton.vcs,
        baton_change_id=baton.change_id,
    )
    write_failover(repo, log)
    return log


def record_outcome(repo: Path, leaf: LeafOutcome) -> FailoverLog:
    """Append one drained leaf to the active window and persist immediately. Tolerates a missing
    window (creates a minimal one) so a mid-window crash never loses the next result."""
    log = read_failover(repo) or FailoverLog(started_at=_now())
    log.outcomes.append(leaf)
    write_failover(repo, log)
    return log


def end_failover(repo: Path) -> FailoverLog | None:
    """Stamp the active window closed. Returns the closed log, or ``None`` if there was none."""
    log = read_failover(repo)
    if log is None:
        return None
    log.ended_at = _now()
    write_failover(repo, log)
    return log


def archive_failover(repo: Path) -> Path | None:
    """Consume the active window: move it to ``history/`` and clear the active file. Switch-back
    calls this once it has reconciled, so ``status`` reflects "no active window" afterwards and the
    next failover starts clean. Returns the history path, or ``None`` if there was no window."""
    log = read_failover(repo)
    if log is None:
        return None
    _archive(repo, log)
    failover_path(repo).unlink(missing_ok=True)
    stamp = re.sub(r"[^0-9A-Za-z-]", "-", log.started_at) or _now()
    return history_dir(repo) / f"{stamp}.json"


def render_failover_summary(log: FailoverLog) -> str:
    """The "while you were away" summary — what the fleet landed, reverted, or skipped, and where to
    look on switch-back. Switch-back reuses this verbatim; the CLI prints it at window close."""
    window = f"{log.started_at} → {log.ended_at or 'open'}"
    head = (
        f"Failover window: {window}"
        + (f"  ·  goal: {log.baton_goal}" if log.baton_goal else "")
        + f"\nDrained {len(log.outcomes)} leaf(s) via tier '{log.model_tier or '?'}': "
        f"{len(log.done)} done · {len(log.failed)} failed · {len(log.skipped)} skipped."
    )
    lines = [head]
    if log.done:
        lines.append("\nLanded (Done):")
        for o in log.done:
            files = f"  [{len(o.changed_files)} file(s)]" if o.changed_files else ""
            commit = f"  {o.commit_id}" if o.commit_id else ""
            lines.append(f"  - {o.project} / {o.task}{commit}{files}")
    if log.failed:
        lines.append("\nFailed (reverted, back to Ready):")
        for o in log.failed:
            lines.append(f"  - {o.project} / {o.task} — {o.reason or 'no reason recorded'}")
    if log.skipped:
        lines.append("\nSkipped:")
        for o in log.skipped:
            lines.append(f"  - {o.project} / {o.task} — {o.reason or 'no reason recorded'}")
    anchor = (
        f"\nSwitch back: read .forge/baton.md, reconcile the landed commits above, and "
        f"`jj diff` the home repo since {log.baton_change_id}."
        if log.baton_change_id
        else "\nSwitch back: read .forge/baton.md and reconcile the landed commits above."
    )
    lines.append(anchor)
    return "\n".join(lines)
