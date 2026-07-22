"""The candidate feed — the scanners' output, and the weekly synthesis's input.

Scanners write here, not into the blip store. A :class:`FeedEntry` is a raw, judgment-free signal
(source, title, url, popularity, first/last seen); the synthesis reads the feed, does the entity
extraction / relevance / quadrant / ring judgment, and *then* creates or moves blips. Keeping the
two apart is what lets messy headline sources (Hacker News) feed the radar without polluting it with
badly-named blips.

Two stores, mirroring the blip store:

- :class:`JsonlCandidateStore` — the canonical local feed at ``.forge/radar/candidates.jsonl``, one
  entry per line (append-shaped, unlike the small whole-file blip store).
- :class:`NousCandidateStore` — the feed projected into a "Radar Candidates" database in the same
  "AI Radar" notebook, so the feed is visible next to the radar.

Dedup is two-layered and both layers live in :mod:`forge.radar.harvest`: exact within-source dedup
by :attr:`FeedEntry.key` (``source:external_id`` — the same item re-fetched each week just bumps
``last_seen``/``times_seen``), and dedup against the blip store by title slug (an item already
promoted to a blip is not re-added to the feed).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from forge.radar.models import Quadrant, slugify
from forge.radar.sources.base import RawItem

DEFAULT_JSONL_PATH = ".forge/radar/candidates.jsonl"
CANDIDATE_DATABASE_TITLE = "Radar Candidates"


class FeedEntry(BaseModel):
    """One candidate signal in the feed: a :class:`RawItem` plus feed bookkeeping."""

    source: str
    external_id: str
    title: str
    url: str
    summary: str = ""
    quadrant_hint: Quadrant | None = None
    score: float | None = None
    published: str | None = None

    first_seen: str  #: ISO date the scanner first surfaced this item.
    last_seen: str  #: ISO date it was most recently surfaced.
    times_seen: int = 1  #: How many scans have surfaced it — a cheap durability/popularity signal.

    @property
    def key(self) -> str:
        """Exact within-source dedup key."""
        return f"{self.source}:{self.external_id}"

    @property
    def slug(self) -> str:
        """Title slug, for dedup against the blip store."""
        return slugify(self.title)

    @classmethod
    def from_item(cls, item: RawItem, *, today: str) -> FeedEntry:
        """A fresh feed entry from a first-seen :class:`RawItem`."""
        return cls(
            source=item.source,
            external_id=item.external_id,
            title=item.title,
            url=item.url,
            summary=item.summary,
            quadrant_hint=item.quadrant_hint,
            score=item.score,
            published=item.published,
            first_seen=today,
            last_seen=today,
            times_seen=1,
        )

    def refreshed(self, item: RawItem, *, today: str) -> FeedEntry:
        """This entry seen again: bump ``last_seen``/``times_seen`` and take the item's latest score
        and summary (both drift over time), preserving ``first_seen``."""
        return self.model_copy(
            update={
                "last_seen": today,
                "times_seen": self.times_seen + 1,
                "score": item.score if item.score is not None else self.score,
                "summary": item.summary or self.summary,
            }
        )


class CandidateFeed(BaseModel):
    """The accumulated feed. Small enough to hold whole; the JSONL store is line-oriented only for
    cheap appends, not because the feed is unbounded (synthesis prunes promoted/stale entries)."""

    entries: list[FeedEntry] = Field(default_factory=list)

    def by_key(self) -> dict[str, FeedEntry]:
        index: dict[str, FeedEntry] = {}
        for entry in self.entries:
            index.setdefault(entry.key, entry)
        return index

    def upsert(self, entry: FeedEntry) -> None:
        """Insert *entry*, or replace the one with the same key in place (preserving order)."""
        for i, existing in enumerate(self.entries):
            if existing.key == entry.key:
                self.entries[i] = entry
                return
        self.entries.append(entry)


@runtime_checkable
class CandidateStore(Protocol):
    """A durable home for the candidate feed. Load returns an empty feed when nothing is stored."""

    def load(self) -> CandidateFeed: ...

    def save(self, feed: CandidateFeed) -> None: ...


class JsonlCandidateStore:
    """The canonical local feed: one :class:`FeedEntry` JSON object per line, written atomically."""

    def __init__(self, path: Path | str = DEFAULT_JSONL_PATH) -> None:
        self.path = Path(path)

    def load(self) -> CandidateFeed:
        if not self.path.is_file():
            return CandidateFeed()
        entries = [
            FeedEntry.model_validate_json(line)
            for line in self.path.read_text().splitlines()
            if line.strip()
        ]
        return CandidateFeed(entries=entries)

    def save(self, feed: CandidateFeed) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = "\n".join(entry.model_dump_json() for entry in feed.entries)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(body + ("\n" if body else ""))
        tmp.replace(self.path)


# --- Nous backend -----------------------------------------------------------

#: Own deterministic namespace (see the blip store's rationale): stable property/row ids so a save
#: is an idempotent update, not a churn.
_NS = uuid.UUID("a11d0001-0000-5000-8000-000000000000")

#: (column name, Nous type). All-text-plus-date, for the same reason as the blip store: the daemon's
#: ``create_database`` mangles caller-supplied select options.
CANDIDATE_COLUMNS: list[tuple[str, str]] = [
    ("Title", "text"),
    ("Source", "text"),
    ("External ID", "text"),
    ("URL", "text"),
    ("Summary", "text"),
    ("Quadrant Hint", "text"),
    ("Score", "text"),
    ("Published", "date"),
    ("First Seen", "date"),
    ("Last Seen", "date"),
    ("Times Seen", "text"),
]


def _stable_id(*parts: str) -> str:
    return str(uuid.uuid5(_NS, ":".join(parts)))


def candidate_database_properties() -> list[dict]:
    """Property definitions for a fresh "Radar Candidates" database, with deterministic ids."""
    return [
        {"id": _stable_id("prop", name), "name": name, "type": ptype}
        for name, ptype in CANDIDATE_COLUMNS
    ]


def _ensure_properties(existing: list[dict]) -> list[dict]:
    by_name = {p["name"].lower(): p for p in existing}
    out: list[dict] = []
    for name, ptype in CANDIDATE_COLUMNS:
        prior = by_name.get(name.lower())
        out.append(
            {"id": prior["id"] if prior else _stable_id("prop", name), "name": name, "type": ptype}
        )
    return out


def _cell(row: dict, prop_map: dict[str, dict], name: str) -> str:
    prop = prop_map.get(name.lower())
    if prop is None:
        return ""
    raw = row.get("cells", {}).get(prop["id"], "")
    return "" if raw is None else str(raw)


def _entry_to_row(entry: FeedEntry, prop_ids: dict[str, str], existing_row_id: str | None) -> dict:
    cells = {
        prop_ids["Title"]: entry.title,
        prop_ids["Source"]: entry.source,
        prop_ids["External ID"]: entry.external_id,
        prop_ids["URL"]: entry.url,
        prop_ids["Summary"]: entry.summary,
        prop_ids["Quadrant Hint"]: entry.quadrant_hint.value if entry.quadrant_hint else "",
        prop_ids["Score"]: "" if entry.score is None else str(entry.score),
        prop_ids["Published"]: entry.published or "",
        prop_ids["First Seen"]: entry.first_seen,
        prop_ids["Last Seen"]: entry.last_seen,
        prop_ids["Times Seen"]: str(entry.times_seen),
    }
    return {"id": existing_row_id or _stable_id("row", entry.key), "cells": cells}


def _row_to_entry(row: dict, prop_map: dict[str, dict]) -> FeedEntry:
    def quad(value: str) -> Quadrant | None:
        try:
            return Quadrant(value) if value else None
        except ValueError:
            return None

    def as_float(value: str) -> float | None:
        try:
            return float(value) if value else None
        except ValueError:
            return None

    def as_int(value: str, default: int = 1) -> int:
        try:
            return int(value) if value else default
        except ValueError:
            return default

    return FeedEntry(
        source=_cell(row, prop_map, "Source"),
        external_id=_cell(row, prop_map, "External ID"),
        title=_cell(row, prop_map, "Title"),
        url=_cell(row, prop_map, "URL"),
        summary=_cell(row, prop_map, "Summary"),
        quadrant_hint=quad(_cell(row, prop_map, "Quadrant Hint")),
        score=as_float(_cell(row, prop_map, "Score")),
        published=_cell(row, prop_map, "Published") or None,
        first_seen=_cell(row, prop_map, "First Seen"),
        last_seen=_cell(row, prop_map, "Last Seen"),
        times_seen=as_int(_cell(row, prop_map, "Times Seen")),
    )


class NousCandidateStore:
    """The candidate feed projected into a "Radar Candidates" Nous database. Whole-database get/put
    on the inner ``database`` object; rows keyed by ``source:external_id`` so saves update in
    place."""

    def __init__(self, client, notebook_id: str, db_id: str) -> None:
        self.client = client
        self.notebook_id = notebook_id
        self.db_id = db_id

    def _inner(self) -> dict:
        content = self.client.get_database(self.notebook_id, self.db_id)
        return content.get("database", content)

    def load(self) -> CandidateFeed:
        db = self._inner()
        prop_map = {p["name"].lower(): p for p in db.get("properties", [])}
        return CandidateFeed(entries=[_row_to_entry(r, prop_map) for r in db.get("rows", [])])

    def save(self, feed: CandidateFeed) -> None:
        db = self._inner()
        properties = _ensure_properties(db.get("properties", []))
        prop_ids = {p["name"]: p["id"] for p in properties}
        existing = _existing_row_ids(db.get("rows", []), prop_map=prop_ids)
        rows = [_entry_to_row(entry, prop_ids, existing.get(entry.key)) for entry in feed.entries]
        db["properties"] = properties
        db["rows"] = rows
        self.client.put_database(self.notebook_id, self.db_id, db)


def _existing_row_ids(rows: list[dict], prop_map: dict[str, str]) -> dict[str, str]:
    """Map ``source:external_id -> existing row id`` so saves update rows in place."""
    src_id, ext_id = prop_map.get("Source"), prop_map.get("External ID")
    if not src_id or not ext_id:
        return {}
    out: dict[str, str] = {}
    for row in rows:
        cells = row.get("cells", {})
        key = f"{cells.get(src_id, '')}:{cells.get(ext_id, '')}"
        if row.get("id") and cells.get(src_id):
            out[key] = row["id"]
    return out


def provision_candidate_store(
    client,
    *,
    notebook_name: str,
    database_title: str = CANDIDATE_DATABASE_TITLE,
    create: bool = True,
) -> NousCandidateStore | None:
    """Find-or-create the "Radar Candidates" database inside the *notebook_name* notebook (the same
    "AI Radar" notebook the blip store uses). Idempotent; ``create=False`` returns ``None`` when
    absent."""
    from forge.radar.store import _find_by_name

    notebook = _find_by_name(client.list_notebooks(), notebook_name, key="name")
    if notebook is None:
        if not create:
            return None
        notebook = client.create_notebook(notebook_name)
    notebook_id = notebook["id"]

    database = _find_by_name(client.list_databases(notebook_id), database_title, key="title")
    if database is None:
        if not create:
            return None
        database = client.create_database(
            notebook_id, database_title, candidate_database_properties()
        )

    return NousCandidateStore(client, notebook_id, database["id"])
