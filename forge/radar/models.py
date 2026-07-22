"""The radar domain: quadrants, rings, blips, and the radar container.

Everything here is plain data (pydantic) with no I/O and no movement logic — the discipline that
governs how a blip changes rings lives in :mod:`forge.radar.movement`, and persistence lives in
:mod:`forge.radar.store`. Dates are ISO-8601 ``YYYY-MM-DD`` strings: the radar is a weekly-cadence
artifact, so day granularity is the right resolution and keeps the Nous database cells simple.
"""

from __future__ import annotations

import re
import unicodedata
from enum import StrEnum

from pydantic import BaseModel, Field


class Quadrant(StrEnum):
    """The four radar quadrants. Centred on *techniques*, not models — model-watching is already
    covered by the scheduled scanner, so the higher-value axis is what changes *how we build*."""

    MODELS = "Models"
    AGENTS = "Agents & Frameworks"
    TECHNIQUES = "Techniques"
    INFRA = "Infra/Tooling"


class Ring(StrEnum):
    """The four adoption rings, outer verdict included. Order matters for movement (see
    :data:`RING_ORDER`): ``Adopt`` is the innermost, highest-confidence ring and ``Hold`` is the
    outer "park it / avoid" verdict."""

    ADOPT = "Adopt"
    TRIAL = "Trial"
    ASSESS = "Assess"
    HOLD = "Hold"


#: Rings from the centre outward. Index 0 (``Adopt``) is the most-adopted; a *promotion* lowers the
#: index (moves toward the centre). ``Hold`` sits at the edge but is treated specially by the
#: movement rules — it is a verdict you can reach from anywhere, not just the ring adjacent to it.
RING_ORDER: list[Ring] = [Ring.ADOPT, Ring.TRIAL, Ring.ASSESS, Ring.HOLD]

#: Where a freshly-seen candidate enters the radar: you *assess* something before you trial it.
DEFAULT_ENTRY_RING: Ring = Ring.ASSESS


def ring_index(ring: Ring) -> int:
    """The centre-outward index of *ring* (``Adopt`` = 0 … ``Hold`` = 3)."""
    return RING_ORDER.index(ring)


def slugify(name: str) -> str:
    """A stable identity key for a blip, derived from its display name.

    Curation depends on a re-surfaced candidate mapping to the *same* blip rather than spawning a
    duplicate, so identity must be stable across capitalisation, punctuation, and accent noise. The
    slug is lowercase ASCII with runs of non-alphanumerics collapsed to single hyphens.
    """
    normalized = unicodedata.normalize("NFKD", name)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_only.lower()).strip("-")
    return slug


class Evidence(BaseModel):
    """One dated observation about a blip — the trail that justifies where it sits and why it moved.

    Evidence **accretes**: writes union new entries onto the existing trail (deduped by
    ``date`` + ``note``) rather than replacing it, so the reasoning history is never silently lost.
    """

    date: str  #: ISO-8601 ``YYYY-MM-DD`` when the observation was recorded.
    note: str  #: What was observed — a benchmark result, a release, a hands-on trial outcome.
    source: str = ""  #: Optional provenance (URL, "hands-on", the scanner that surfaced it).

    def key(self) -> tuple[str, str]:
        """The dedup identity used when accreting evidence."""
        return (self.date, self.note.strip())


class Blip(BaseModel):
    """One item on the radar: a model, framework, technique, or tool, with its current ring and the
    accreted trail of why it sits there.

    ``rationale`` and ``action`` are *current state* — the one-line "why it's here now" and "what to
    do about it" — and are meant to be rewritten each synthesis cycle. ``evidence`` and ``links``
    accrete. ``ring_last`` / ``last_moved`` are the movement bookkeeping the anti-thrash rules read.
    """

    name: str  #: Display name, e.g. "Qwen3-Coder 30B" or "Structured tool-calling".
    quadrant: Quadrant
    ring: Ring

    ring_last: Ring | None = None  #: The ring before the most recent move; ``None`` if never moved.
    first_seen: str  #: ISO-8601 date the blip first entered the radar.
    last_seen: str  #: ISO-8601 date a scanner most recently surfaced it (freshness signal).
    last_moved: str | None = (
        None  #: ISO-8601 date of the most recent ring change; ``None`` if none.
    )

    rationale: str = ""  #: Current one-line justification for the ring (rewritten each cycle).
    action: str = ""  #: Current recommended action, if any ("trial on the euclid router").
    evidence: list[Evidence] = Field(default_factory=list)  #: Accreting dated trail.
    links: list[str] = Field(default_factory=list)  #: Accreting source URLs.

    @property
    def slug(self) -> str:
        """Stable identity key (see :func:`slugify`)."""
        return slugify(self.name)

    def moved(self) -> bool:
        """Whether this blip has ever changed rings since first seen."""
        return self.ring_last is not None


class Radar(BaseModel):
    """The whole radar: every blip, plus small query helpers. Small by nature (dozens of blips), so
    the store loads and saves it whole rather than diffing rows."""

    blips: list[Blip] = Field(default_factory=list)

    def by_slug(self) -> dict[str, Blip]:
        """Index of blips keyed by :func:`slugify` slug. On the (guarded-against) chance of a
        duplicate slug, the first occurrence wins."""
        index: dict[str, Blip] = {}
        for blip in self.blips:
            index.setdefault(blip.slug, blip)
        return index

    def get(self, name_or_slug: str) -> Blip | None:
        """The blip matching *name_or_slug* (by slug), or ``None``."""
        return self.by_slug().get(slugify(name_or_slug))

    def upsert(self, blip: Blip) -> None:
        """Insert *blip*, or replace the existing blip with the same slug in place (preserving
        position so the radar's row order is stable across writes)."""
        slug = blip.slug
        for i, existing in enumerate(self.blips):
            if existing.slug == slug:
                self.blips[i] = blip
                return
        self.blips.append(blip)

    def counts(self) -> dict[Quadrant, dict[Ring, int]]:
        """Blip counts per quadrant × ring — the shape the status view and renderer read."""
        grid: dict[Quadrant, dict[Ring, int]] = {q: {r: 0 for r in RING_ORDER} for q in Quadrant}
        for blip in self.blips:
            grid[blip.quadrant][blip.ring] += 1
        return grid
