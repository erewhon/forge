"""Scan direct dependencies for newer available versions.

Produces ``BumpCandidate`` lists from ``uv tree --outdated`` output, with
version-delta classification (patch / minor / major / unknown) and capped,
sorted candidate lists ready for downstream gates.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from agents.dependabot.config import settings
from agents.dependabot.models import BumpCandidate, DeltaClass


class ScanError(RuntimeError):
    """The ``uv tree --outdated`` invocation failed."""


# Matches lines of the form:
#   ├── httpx v0.28.1
#   ├── anthropic v0.79.0 (latest: v0.116.0)
#   └── pytest v9.0.2 (group: dev) (latest: v9.1.1)
# The regex ignores leading tree-drawing characters, groups, and other
# parenthesised metadata, capturing only the package name, current version,
# and the latest version (when present).
_LINE_RE = re.compile(
    r"^[\s├─└│]*"  # optional leading tree-drawing characters
    r"(?P<name>\S+)"  # package name
    r"\s+v(?P<current>\S+)"  # current version (with leading 'v')
    r"(?:\s+\(.*?\))*"  # optional parenthesised metadata (group, etc.)
    r"\s+\(latest:\s+v(?P<latest>\S+)\)"  # required latest version suffix
)


def classify_delta(current: str, latest: str) -> DeltaClass:
    """Classify the semantic-version delta between *current* and *latest*.

    Strips leading ``v``, splits on ``.``, and compares numerically from left
    to right:

    - major differs (first segment)  -> ``"major"``
    - minor differs (second segment) -> ``"minor"``
    - patch differs (third segment)  -> ``"patch"``
    - any non-integer segment or missing part -> ``"unknown"``

    ``"unknown"`` is treated like ``"major"`` downstream (never auto-merged).
    """
    cur = current.lstrip("v")
    lat = latest.lstrip("v")
    parts_cur = cur.split(".")
    parts_lat = lat.split(".")

    # All parts must be integer-parseable.
    try:
        nums_cur = [int(p) for p in parts_cur]
        nums_lat = [int(p) for p in parts_lat]
    except ValueError:
        return "unknown"

    # Different number of segments is treated as unknown (e.g. "1.2.3" vs "2024.1").
    if len(parts_cur) != len(parts_lat):
        return "unknown"

    # Compare from most significant to least.
    for i, (a, b) in enumerate(zip(nums_cur, nums_lat)):
        if a != b:
            if i == 0:
                return "major"
            if i == 1:
                return "minor"
            return "patch"

    return "patch"  # identical — treat as patch (no-op downstream)


_DELTA_ORDER: dict[DeltaClass, int] = {
    "patch": 0,
    "minor": 1,
    "major": 2,
    "unknown": 3,
}


def scan_outdated(
    repo: Path,
    *,
    timeout: int | None = None,
) -> list[BumpCandidate]:
    """Run ``uv tree --outdated --depth 1 --no-dedupe`` and return BumpCandidates.

    Only direct dependencies are scanned (``--depth 1``).  Lines without a
    ``(latest: vX.Y.Z)`` suffix are silently skipped.

    Non-zero exit raises :class:`ScanError` (stderr included in the message).
    Never returns ``[]`` silently — if ``uv`` fails, an error is raised.

    Results are sorted: ``patch`` first, then ``minor``, then ``major``, then
    ``unknown``, alphabetically within each class.  The list is capped at
    ``settings.max_candidates``.
    """
    timeout = timeout if timeout is not None else settings.scan_timeout

    result = subprocess.run(
        ["uv", "tree", "--outdated", "--depth", "1", "--no-dedupe"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise ScanError(
            f"uv tree --outdated failed (exit {result.returncode}): {result.stderr.strip()}"
        )

    candidates: list[BumpCandidate] = []
    for line in result.stdout.splitlines():
        m = _LINE_RE.match(line.strip())
        if not m:
            continue
        name = m.group("name")
        current = m.group("current")
        latest = m.group("latest")
        delta = classify_delta(current, latest)
        candidates.append(BumpCandidate(name=name, current=current, latest=latest, delta=delta))

    # Sort: patch -> minor -> major -> unknown; alphabetical within class.
    candidates.sort(key=lambda c: (_DELTA_ORDER[c.delta], c.name))

    return candidates[: settings.max_candidates]
