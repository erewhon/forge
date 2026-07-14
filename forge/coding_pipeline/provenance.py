"""Repo-native leaf provenance: a git note on every landed ``auto:`` commit recording how the
machine produced it — task, external_ref, spec hash, model tier, changed files, duration, and the
wave it landed in. ``git log --notes=pipeline/provenance`` becomes the audit of which commits were
model-written and under what task.

**Written after the reconcile barrier, never at worker-commit time.** Concurrent-worker commits
are rebased during the serial reconcile, so workspace-time commit ids die; leaves are addressed by
jj *change id* (stable across rebases), which is what ``LeafOutcome.commit_id`` carries on the jj
path. This module resolves that change id to the CURRENT git commit sha — i.e. the post-rebase one
— and attaches the note there. On the git path ``commit_id`` is already a sha.

Provenance is best-effort: a resolution or write failure warns and is skipped, and never disturbs
the wave. Reuses :mod:`forge.shared.git_notes`; payload shapes stay pipeline-specific here.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Callable
from pathlib import Path

from forge.coding_pipeline.models import LeafOutcome
from forge.shared.git_notes import GitError, write_note
from forge.task_worker.models import TaskInfo
from forge.task_worker.vcs import detect_vcs

#: Note namespace for per-leaf provenance; git stores it at ``refs/notes/pipeline/provenance``.
LEAF_NOTE_REF = "pipeline/provenance"

_TIMEOUT = 30


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT, cwd=cwd)


def spec_sha256(spec: str) -> str:
    """Content hash of the spec markdown a worker received — lets an auditor verify which spec
    version produced a commit. Prefixed with the algorithm for forward compatibility."""
    return "sha256:" + hashlib.sha256(spec.encode("utf-8")).hexdigest()


def resolve_git_commit(repo: Path, commit_id: str) -> str | None:
    """Resolve a landed leaf's ``commit_id`` to a git commit sha a note can attach to, or ``None``
    when it cannot be resolved.

    On the jj path ``commit_id`` is a change id (stable across the reconcile rebase); translate it
    to the change's CURRENT git sha via jj's ``commit_id`` template. On the git path it is already
    a sha; verify it still resolves to a commit object.
    """
    if not commit_id:
        return None
    if detect_vcs(repo) == "jj":
        res = _run(
            ["jj", "log", "-r", commit_id, "--no-graph", "-T", "commit_id", "--limit", "1"], repo
        )
        return res.stdout.strip() or None
    res = _run(["git", "rev-parse", "--verify", "--quiet", f"{commit_id}^{{commit}}"], repo)
    return res.stdout.strip() or None


def leaf_note_payload(
    outcome: LeafOutcome,
    task: TaskInfo | None,
    *,
    wave: int,
    spec_sha: str | None,
    timestamp: str,
) -> dict:
    """Build the versioned-JSON provenance record for one landed leaf.

    Sourced from the ``LeafOutcome`` (what landed), the ``TaskInfo`` (how it was routed), the wave
    number, and the spec hash. ``task`` may be ``None`` (leaf not found in Forge at note time) — the
    routing fields are then omitted rather than guessed. No secrets: task metadata only.
    """
    payload: dict = {
        "schema": 1,
        "kind": "leaf",
        "leaf": outcome.leaf,
        "spec_sha256": spec_sha,
        "changed_files": list(outcome.changed_files),
        "duration_s": outcome.duration_s,
        "wave": wave,
        "timestamp": timestamp,
    }
    if task is not None:
        payload.update(
            external_ref=task.external_ref,
            project=task.project,
            model_tier=task.model_tier,
            task_type=task.task_type,
            complexity=task.complexity,
            requires_tests=task.requires_tests,
        )
    return payload


def write_leaf_note(
    repo: Path,
    outcome: LeafOutcome,
    task: TaskInfo | None,
    *,
    wave: int,
    spec_sha: str | None,
    timestamp: str,
    log: Callable[[str], None] = print,
) -> str | None:
    """Attach the leaf provenance note to the landed commit; return the git sha it landed on, or
    ``None`` when nothing was written (unresolvable commit, or git refused). Best-effort — the
    caller supplies ``timestamp`` (this module does not read the clock) and never fails on our
    account. Pushing rides the existing ``refs/notes/pipeline/*`` checkpoint wiring."""
    sha = resolve_git_commit(repo, outcome.commit_id or "")
    if sha is None:
        log(f"warning: cannot resolve commit for leaf {outcome.leaf!r}; skipping provenance note")
        return None
    payload = leaf_note_payload(outcome, task, wave=wave, spec_sha=spec_sha, timestamp=timestamp)
    try:
        write_note(repo, LEAF_NOTE_REF, sha, payload)
    except GitError as exc:
        log(f"warning: writing provenance note for {outcome.leaf!r} failed: {exc}")
        return None
    return sha


def record_leaf_provenance(
    repo: Path,
    outcomes: list[LeafOutcome],
    *,
    find: Callable[[str], TaskInfo | None],
    fetch_spec: Callable[[str], str],
    wave: int,
    timestamp: str,
    log: Callable[[str], None] = print,
) -> None:
    """Write a provenance note for every LANDED leaf in ``outcomes`` (``status == "done"`` with a
    commit id). Called from the dispatch layer AFTER the reconcile barrier, so commit ids are
    final. Failed/skipped/demoted leaves (no commit id) get no note. Every per-leaf lookup and
    write is guarded — provenance must never disturb a completed wave.
    """
    for outcome in outcomes:
        if outcome.status != "done" or not outcome.commit_id:
            continue
        try:
            task = find(outcome.leaf)
        except Exception as exc:  # noqa: BLE001 — a lookup miss must not sink provenance
            log(f"warning: task lookup for provenance note failed ({outcome.leaf!r}): {exc}")
            task = None
        spec_sha: str | None = None
        try:
            spec_sha = spec_sha256(fetch_spec(outcome.leaf))
        except Exception as exc:  # noqa: BLE001 — spec hash is best-effort context
            log(f"warning: spec fetch for provenance hash failed ({outcome.leaf!r}): {exc}")
        write_leaf_note(
            repo, outcome, task, wave=wave, spec_sha=spec_sha, timestamp=timestamp, log=log
        )
