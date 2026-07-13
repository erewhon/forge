"""Clone-portable provenance as git notes: versioned-JSON payloads under ``refs/notes/pipeline/*``.

The coding pipeline's decision artifacts (gate verdicts, leaf provenance) move INTO the target
repo as git objects, so a clone carries permanent, self-contained evidence of what the robot
reviewers decided. Notes are plain ``git notes --ref=<ref>`` — which work identically in a
jj-colocated repo (a colocated repo is a plain git repo underneath; new refs are picked up by
the next jj command). Payloads are versioned JSON (``{"schema": 1, ...}``) so readers can evolve
the shape without breaking old clones. Timestamps are never generated here — callers pass them
in — which keeps the library deterministic and its round-trips testable.

This is the shared plumbing the rest of the Repo-Native Provenance feature builds on
(leaf-provenance notes, the journal mirror); keep it free of pipeline-specific payload shapes.
"""

from __future__ import annotations

import json
from pathlib import Path

from forge.shared.gitops import GitError, git

SCHEMA_VERSION = 1


def write_note(repo: Path, ref: str, commit: str, payload: dict) -> None:
    """Attach *payload* as a JSON git note under ``refs/notes/<ref>`` on *commit*, replacing any
    existing note (``add -f``).

    *ref* is the short note namespace (e.g. ``pipeline/gate``); git stores it at
    ``refs/notes/pipeline/gate``. The body is canonicalised (``sort_keys``) so an unchanged
    payload re-writes byte-identically. Raises :class:`GitError` when git refuses the write
    (e.g. *commit* is not a valid object) — callers that want provenance to be best-effort
    should catch it and warn rather than fail.
    """
    body = json.dumps(payload, sort_keys=True)
    git(repo, "notes", f"--ref={ref}", "add", "-f", "-m", body, commit)


def read_note(repo: Path, ref: str, commit: str) -> dict | None:
    """Return the JSON payload of the note under ``refs/notes/<ref>`` on *commit*, or ``None``
    when no such note exists.

    A note whose body is not valid JSON raises :class:`ValueError` — a malformed provenance
    record is a bug worth surfacing, not silently dropping.
    """
    try:
        body = git(repo, "notes", f"--ref={ref}", "show", commit)
    except GitError:
        # No note for this object (the common case), or an unreadable ref — either way there is
        # no payload to return. Callers treat absence as "not recorded".
        return None
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"note {ref} on {commit} is not valid JSON: {exc}") from exc
