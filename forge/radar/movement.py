"""The curation discipline — the hard part of a *living* radar.

A radar's worth is the movement of a blip and the reason for it, not a weekly re-dump. That makes
curation a state-machine problem with two deliberately separated mechanisms:

- **Candidates accumulate** — scanners surface the same thing over and over, noisily.
  :func:`integrate_candidate` folds a candidate into its existing blip (refresh ``last_seen``,
  accrete evidence and links) or creates it at the entry ring. It **never moves a ring**: seeing a
  thing more often is not evidence it should be adopted.
- **Blips move rarely, and only with evidence** — the weekly synthesis is the only thing that
  changes a ring, via :func:`propose_move`, which enforces the anti-thrash rules below.

Anti-thrash rules (all bypassable with ``force=True`` — the explicit override):

1. **Movement needs evidence.** A move must carry a non-empty ``rationale``; it is recorded as a
   dated :class:`~forge.radar.models.Evidence` entry so the *why* is never lost.
2. **Cooldown.** A blip that moved within ``cooldown_days`` does not move again — churn is the
   failure mode a radar is most prone to.
3. **No reversal.** Moving a blip straight back to the ring it just came from (``ring_last``) is the
   signature of thrash and is refused within the cooldown even if rule 2 would otherwise pass.
4. **Step toward the centre one ring at a time.** Promotions run Assess → Trial → Adopt; you cannot
   jump Assess → Adopt without ``allow_jump=True``. ``Hold`` is exempt — it is a verdict reachable
   from anywhere (park/deprecate), and leaving ``Hold`` re-enters at the adjacent ring unless a jump
   is allowed.

All functions here are pure: they mutate the passed-in :class:`~forge.radar.models.Radar`/blips and
take an explicit ``today`` so tests are deterministic. Persistence is the store's job.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from forge.radar.models import (
    DEFAULT_ENTRY_RING,
    Blip,
    Evidence,
    Quadrant,
    Radar,
    Ring,
    ring_index,
)

#: Default churn guard: a blip that moved fewer than this many days ago will not move again without
#: ``force=True``. One week matches the radar's weekly synthesis cadence — at most one move per blip
#: per cycle.
DEFAULT_COOLDOWN_DAYS = 7


def _iso(day: date) -> str:
    return day.isoformat()


def _accrete_evidence(blip: Blip, entries: list[Evidence]) -> None:
    """Union *entries* onto ``blip.evidence`` in place, deduped by :meth:`Evidence.key`, existing
    first — the accretion that keeps the reasoning trail from being silently dropped."""
    seen = {e.key() for e in blip.evidence}
    for entry in entries:
        if entry.key() not in seen:
            blip.evidence.append(entry)
            seen.add(entry.key())


def _accrete_links(blip: Blip, links: list[str]) -> None:
    """Union *links* onto ``blip.links`` in place, order-preserving and deduped."""
    for link in links:
        if link and link not in blip.links:
            blip.links.append(link)


class Candidate(BaseModel):
    """A raw signal from a source scanner: a thing that might belong on the radar. Noisy and
    frequent — the same candidate re-surfaces across scans. Identity is by name (via the blip
    slug)."""

    name: str
    quadrant: Quadrant
    summary: str = ""  #: One-line description; seeds a new blip's rationale, never overwrites one.
    links: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    source: str = (
        ""  #: Which scanner surfaced it (for provenance on the first-seen evidence entry).
    )


class IntegrationResult(BaseModel):
    """Outcome of folding a candidate into the radar."""

    created: bool  #: True if a new blip was added, False if an existing one was refreshed.
    slug: str


def integrate_candidate(
    radar: Radar,
    candidate: Candidate,
    *,
    today: date,
    entry_ring: Ring = DEFAULT_ENTRY_RING,
) -> IntegrationResult:
    """Fold *candidate* into *radar*: create its blip at ``entry_ring`` if new, else refresh the
    existing blip's freshness and accrete its evidence/links. **Never changes a ring** — that is
    :func:`propose_move`'s job. Mutates *radar* in place.
    """
    existing = radar.get(candidate.name)
    stamp = _iso(today)

    if existing is None:
        first_evidence = [Evidence(date=stamp, note="First seen", source=candidate.source)]
        first_evidence.extend(candidate.evidence)
        blip = Blip(
            name=candidate.name,
            quadrant=candidate.quadrant,
            ring=entry_ring,
            first_seen=stamp,
            last_seen=stamp,
            rationale=candidate.summary,
            links=list(candidate.links),
        )
        _accrete_evidence(blip, first_evidence)
        radar.upsert(blip)
        return IntegrationResult(created=True, slug=blip.slug)

    existing.last_seen = stamp
    _accrete_links(existing, candidate.links)
    _accrete_evidence(existing, candidate.evidence)
    return IntegrationResult(created=False, slug=existing.slug)


class MoveDecision(BaseModel):
    """The result of a :func:`propose_move`. ``applied`` says whether the ring actually changed;
    ``kind`` explains why (and, when refused, which rule blocked it)."""

    applied: bool
    kind: str  #: "promote" | "demote" | "hold" | "noop" | "not-found" |
    #: "no-rationale" | "cooldown" | "reversal" | "non-adjacent"
    slug: str
    from_ring: Ring | None = None
    to_ring: Ring | None = None
    reason: str = ""  #: Human-readable explanation (the refusal reason, or the move's rationale).


def _move_kind(from_ring: Ring, to_ring: Ring) -> str:
    if to_ring == Ring.HOLD:
        return "hold"
    if ring_index(to_ring) < ring_index(from_ring):
        return "promote"
    return "demote"


def _is_adjacent_step(from_ring: Ring, to_ring: Ring) -> bool:
    """Whether ``from_ring`` → ``to_ring`` is an allowed single step.

    ``Hold`` is exempt from the adjacency rule in both directions: parking or deprecating a blip is
    a verdict reachable from any ring, and re-entering the pipeline from ``Hold`` lands at the
    adjacent ``Assess``. Between the pipeline rings (Adopt/Trial/Assess) a step moves one index.
    """
    if from_ring == Ring.HOLD or to_ring == Ring.HOLD:
        return True
    return abs(ring_index(from_ring) - ring_index(to_ring)) == 1


def propose_move(
    radar: Radar,
    name_or_slug: str,
    to_ring: Ring,
    rationale: str,
    *,
    today: date,
    source: str = "synthesis",
    force: bool = False,
    allow_jump: bool = False,
    cooldown_days: int = DEFAULT_COOLDOWN_DAYS,
) -> MoveDecision:
    """Attempt a disciplined ring change for the named blip. Applies it and records the bookkeeping
    (``ring_last`` / ``last_moved`` / an accreted evidence entry) only if every rule passes; returns
    a :class:`MoveDecision` describing the outcome either way. Mutates *radar* in place.

    ``force=True`` bypasses the cooldown, reversal, and adjacency guards (but not the
    empty-rationale check — a forced move still has to say why). ``allow_jump=True`` bypasses only
    adjacency.
    """
    blip = radar.get(name_or_slug)
    if blip is None:
        return MoveDecision(
            applied=False,
            kind="not-found",
            slug=name_or_slug,
            reason=f"No blip named {name_or_slug!r}",
        )

    slug = blip.slug
    from_ring = blip.ring

    if not rationale.strip():
        return MoveDecision(
            applied=False,
            kind="no-rationale",
            slug=slug,
            from_ring=from_ring,
            to_ring=to_ring,
            reason="A move must carry a rationale — movement needs evidence.",
        )

    if to_ring == from_ring:
        return MoveDecision(
            applied=False,
            kind="noop",
            slug=slug,
            from_ring=from_ring,
            to_ring=to_ring,
            reason=f"Already in {from_ring.value}.",
        )

    if not force:
        # Reversal: straight back to the ring it just left is the signature of thrash.
        if (
            blip.ring_last is not None
            and to_ring == blip.ring_last
            and _within_cooldown(blip, today, cooldown_days)
        ):
            return MoveDecision(
                applied=False,
                kind="reversal",
                slug=slug,
                from_ring=from_ring,
                to_ring=to_ring,
                reason=(
                    f"Refusing to reverse {from_ring.value} → {to_ring.value} within "
                    f"{cooldown_days}d of the last move (thrash guard). Pass force=True to "
                    "override."
                ),
            )
        if _within_cooldown(blip, today, cooldown_days):
            return MoveDecision(
                applied=False,
                kind="cooldown",
                slug=slug,
                from_ring=from_ring,
                to_ring=to_ring,
                reason=(
                    f"Moved {blip.last_moved} — within the {cooldown_days}d cooldown. "
                    "Pass force=True to override."
                ),
            )

    if not allow_jump and not force and not _is_adjacent_step(from_ring, to_ring):
        return MoveDecision(
            applied=False,
            kind="non-adjacent",
            slug=slug,
            from_ring=from_ring,
            to_ring=to_ring,
            reason=(
                f"{from_ring.value} → {to_ring.value} skips a ring; promotions step one ring "
                "toward the centre. Pass allow_jump=True for a justified jump."
            ),
        )

    # Apply.
    blip.ring_last = from_ring
    blip.ring = to_ring
    blip.last_moved = _iso(today)
    blip.rationale = rationale.strip()
    _accrete_evidence(
        blip,
        [
            Evidence(
                date=_iso(today),
                note=f"{from_ring.value} → {to_ring.value}: {rationale.strip()}",
                source=source,
            )
        ],
    )
    return MoveDecision(
        applied=True,
        kind=_move_kind(from_ring, to_ring),
        slug=slug,
        from_ring=from_ring,
        to_ring=to_ring,
        reason=rationale.strip(),
    )


def _within_cooldown(blip: Blip, today: date, cooldown_days: int) -> bool:
    """Whether *blip* moved within ``cooldown_days`` of *today*. A blip that never moved (or has an
    unparseable ``last_moved``) is not in cooldown."""
    if blip.last_moved is None:
        return False
    try:
        moved = date.fromisoformat(blip.last_moved)
    except ValueError:
        return False
    return (today - moved).days < cooldown_days
