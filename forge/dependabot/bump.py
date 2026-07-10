"""Apply a single BumpCandidate as a working-copy manifest-only change.

Runs ``uv lock --upgrade-package <name>`` which re-resolves that one package
(bounded by ``pyproject.toml`` constraints) and rewrites ``uv.lock`` in place.

Boundedness semantics: *apply_bump* pushes the package as far as the
``pyproject.toml`` constraints allow — which may be short of ``latest`` if the
constraint caps it. Returns ``[]`` when the lock did not change (constraint
already pins it); the caller treats that as a skip, not an error.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from forge.dependabot.models import BumpCandidate
from forge.shared.automerge import working_diff
from forge.task_worker.vcs import (
    VCSError,
    get_changed_files,
    revert_changes,
)


class BumpError(RuntimeError):
    """The ``uv lock`` invocation failed."""


def apply_bump(
    repo: Path,
    candidate: BumpCandidate,
    *,
    timeout: int | None = None,
) -> list[str]:
    """Run ``uv lock --upgrade-package <name>`` and return the changed files.

    On success returns the list of files from
    :func:`forge.task_worker.vcs.get_changed_files`.
    Returns ``[]`` when the lock did not change.

    On non-zero exit from ``uv lock``, calls
    :func:`forge.task_worker.vcs.revert_changes` once and raises
    :class:`BumpError` carrying the stderr text.
    """
    result = subprocess.run(
        ["uv", "lock", "--upgrade-package", candidate.name],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=repo,
    )
    if result.returncode != 0:
        revert_changes(repo)
        raise BumpError(
            f"uv lock --upgrade-package {candidate.name!r} failed (exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return get_changed_files(repo)


def lockfile_delta(repo: Path) -> list[str]:
    """Parse the working diff for ``uv.lock`` and extract version deltas.

    Reads :func:`forge.shared.automerge.working_diff`, looks for ``uv.lock``
    hunks, and extracts ``name old->new`` strings from removed and added
    ``version = "..."`` lines grouped by ``name = "..."``.

    Returns a compact list like ``["foo 1.0.0->1.0.1"]``.
    Returns ``[]`` (best-effort) when the diff is unparseable.
    """
    try:
        diff_text = working_diff(repo)
    except VCSError:
        return []

    return _parse_lockfile_diff(diff_text)


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------

# Matches a uv.lock ``name = "..."`` entry (with optional diff prefix and space).
_NAME_RE = re.compile(r'[-+]? ?name = "([^"]+)"')
# Matches a ``version = "..."`` entry inside a package block.
_VERSION_RE = re.compile(r'[-+]? ?version = "([^"]+)"')
# Hunk header indicating a uv.lock file (git/jj unified diff).
_LOCK_HEADER_RE = re.compile(r"^(?:---|\+\+\+) .*/?uv\.lock")
# Any file header — used to LEAVE the uv.lock section when the diff moves to another file,
# so that file's `version = "..."` lines (e.g. pyproject's project version) can't be
# misattributed to the last lockfile package.
_FILE_HEADER_RE = re.compile(r"^(?:diff --git |--- |\+\+\+ )")


def _parse_lockfile_diff(diff_text: str) -> list[str]:
    """Extract ``name old->new`` pairs from a unified diff of ``uv.lock``.

    Walks the diff tracking the current package *name* and records removed vs
    added *version* lines per name.  A delta is emitted only when both an old
    and a new version appear for the same name.
    """
    deltas: list[str] = []
    in_lock_hunk = False
    # Ordered insertion map: seq_id -> {name, removed: str|None, added: str|None}
    entries: list[dict[str, str | None]] = []

    def _flush() -> None:
        for entry in entries:
            if entry["removed"] and entry["added"]:
                deltas.append(f"{entry['name']} {entry['removed']}->{entry['added']}")
        entries.clear()

    for line in diff_text.splitlines():
        # Detect lock-file hunk header (--- a/uv.lock / +++ b/uv.lock)
        if _LOCK_HEADER_RE.match(line):
            _flush()
            in_lock_hunk = True
            continue
        if _FILE_HEADER_RE.match(line):
            # The diff moved on to a different file — leave the lock section.
            _flush()
            in_lock_hunk = False
            continue

        if not in_lock_hunk:
            continue

        # Try to match name = "..."
        name_m = _NAME_RE.match(line)
        if name_m:
            entries.append({"name": name_m.group(1), "removed": None, "added": None})
            continue

        # Try to match version = "..."
        ver_m = _VERSION_RE.match(line)
        if ver_m and entries:
            last = entries[-1]
            if line.startswith("-"):
                last["removed"] = ver_m.group(1)
            elif line.startswith("+"):
                last["added"] = ver_m.group(1)
            continue

    _flush()  # final hunk
    return deltas
