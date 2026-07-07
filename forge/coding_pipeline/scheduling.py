"""Conflict-free batch picking over architect-predicted file scopes.

Pure optimization, safe by design: the reconcile barrier is the correctness floor, so a
wrong pick costs wasted parallel work (a colliding leaf burns a worker run before detection
reverts it), never corruption. A leaf with an EMPTY scope — replan fix-ups carry none —
overlaps everything: it still dispatches, alone, but never concurrently with a sibling.
Deferred leaves simply stay Ready for the next wave; the dispatcher journals what it
deferred so a shrunken batch is always legible, never a silent cap.
"""

from __future__ import annotations


def _parts(entry: str) -> tuple[str, ...]:
    return tuple(p for p in entry.strip().strip("/").split("/") if p)


def _entries_overlap(a: str, b: str) -> bool:
    """Path-prefix overlap: `agents/shared/` vs `agents/shared/pool.py` collide; so do
    identical entries. Distinct siblings (`a/x.py` vs `a/y.py`) do not."""
    pa, pb = _parts(a), _parts(b)
    if not pa or not pb:
        return True  # a degenerate entry ("", "/") claims everything — conservative
    n = min(len(pa), len(pb))
    return pa[:n] == pb[:n]


def scopes_overlap(a: list[str], b: list[str]) -> bool:
    """Empty scope = unknown = overlaps everything (fix-ups, legacy trees)."""
    if not a or not b:
        return True
    return any(_entries_overlap(x, y) for x in a for y in b)


def pick_disjoint(leaves: list[tuple[str, list[str]]]) -> tuple[list[str], list[str]]:
    """Greedy split of ``(title, file_scope)`` pairs into (batch, deferred), input order
    preserved — dispatch order is priority order, so earlier leaves win contested scopes.

    Guarantee: a non-empty input always yields a non-empty batch (the head leaf dispatches
    even with an empty scope — alone is safe; an empty BATCH would deadlock the wave).
    """
    batch: list[tuple[str, list[str]]] = []
    deferred: list[str] = []
    for title, scope in leaves:
        if not batch:
            batch.append((title, scope))
            continue
        if not scope or any(scopes_overlap(scope, taken) for _, taken in batch):
            deferred.append(title)
            continue
        batch.append((title, scope))
    return [t for t, _ in batch], deferred
