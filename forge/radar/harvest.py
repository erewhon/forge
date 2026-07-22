"""Harvest — run the source adapters, filter the firehose, and accumulate the candidate feed.

The one orchestration step of the scanner layer. For each adapter it fetches items, drops the
irrelevant ones (the cheap gate), drops the ones already promoted to blips, and folds the rest into
the feed: a re-seen item bumps its ``last_seen``/``times_seen``, a fresh one is appended with a
provisional quadrant hint. No blips are created or moved here — that is the synthesis's job. A
:class:`HarvestReport` records what each source contributed so a scan is auditable and a dark source
is visible rather than silent.
"""

from __future__ import annotations

from datetime import date

import httpx
from pydantic import BaseModel, Field

from forge.radar.candidates import CandidateFeed, FeedEntry
from forge.radar.models import slugify
from forge.radar.relevance import classify_quadrant, is_relevant
from forge.radar.sources.base import RawItem, SourceAdapter


class SourceReport(BaseModel):
    """What one adapter contributed to a harvest."""

    source: str
    fetched: int = 0  #: Items the adapter returned.
    dropped_irrelevant: int = 0  #: Failed the relevance gate.
    skipped_blip: int = 0  #: Already on the radar as a blip — not re-added to the feed.
    refreshed: int = 0  #: Already in the feed — last_seen/times_seen bumped.
    added: int = 0  #: New feed entries.
    error: str = ""  #: Non-empty when the source failed to fetch (the source went dark).


class HarvestReport(BaseModel):
    sources: list[SourceReport] = Field(default_factory=list)

    @property
    def added(self) -> int:
        return sum(s.added for s in self.sources)

    def render(self) -> str:
        lines = ["source          fetched  kept  new  refreshed  dropped"]
        for s in self.sources:
            if s.error:
                lines.append(f"{s.source:<15} ERROR: {s.error}")
                continue
            kept = s.added + s.refreshed
            lines.append(
                f"{s.source:<15} {s.fetched:>7}  {kept:>4}  {s.added:>3}  "
                f"{s.refreshed:>9}  {s.dropped_irrelevant + s.skipped_blip:>7}"
            )
        lines.append("")
        lines.append(f"{self.added} new candidate(s) added to the feed.")
        return "\n".join(lines)


def _text(item: RawItem) -> str:
    return f"{item.title} {item.summary}".strip()


def harvest(
    feed: CandidateFeed,
    adapters: list[SourceAdapter],
    client: httpx.Client,
    *,
    blip_slugs: set[str],
    today: date,
) -> HarvestReport:
    """Run *adapters*, folding survivors into *feed* (mutated in place). *blip_slugs* is
    ``radar.by_slug().keys()`` — items whose title already matches a blip are skipped. Returns a
    per-source :class:`HarvestReport`."""
    stamp = today.isoformat()
    index = feed.by_key()
    report = HarvestReport()

    for adapter in adapters:
        sr = SourceReport(source=adapter.name)
        try:
            items = adapter.fetch(client)
        except Exception as exc:  # a dark source must not abort the whole harvest
            sr.error = f"{type(exc).__name__}: {exc}"
            report.sources.append(sr)
            continue

        sr.fetched = len(items)
        for item in items:
            text = _text(item)
            if not is_relevant(text):
                sr.dropped_irrelevant += 1
                continue
            if slugify(item.title) in blip_slugs:
                sr.skipped_blip += 1
                continue

            existing = index.get(item.key)
            if existing is not None:
                updated = existing.refreshed(item, today=stamp)
                feed.upsert(updated)
                index[item.key] = updated
                sr.refreshed += 1
                continue

            if item.quadrant_hint is None:
                item = item.model_copy(update={"quadrant_hint": classify_quadrant(text)})
            entry = FeedEntry.from_item(item, today=stamp)
            feed.upsert(entry)
            index[entry.key] = entry
            sr.added += 1

        report.sources.append(sr)

    return report
