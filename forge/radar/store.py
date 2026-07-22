"""Persistence for the radar, behind a small :class:`RadarStore` protocol.

The radar is small (dozens of blips) and curation reads/writes it whole, so a store is just
``load() -> Radar`` and ``save(radar)`` — no row diffing. Two backends:

- :class:`JsonRadarStore` — a durable local JSON file (``.forge/radar/blips.json`` by default),
  written atomically. This is the canonical, lossless store and the one the tests exercise.
- :class:`NousRadarStore` — projects the radar into the "AI Radar" Nous notebook's blip database,
  one row per blip, so it is visible and shareable in Nous. It round-trips through the daemon's
  whole-database ``get``/``put`` and reuses existing property/row ids so repeated saves are
  idempotent.

The Nous mapping (schema, cell (de)serialisation) is kept pure and duck-typed on a tiny client
protocol so it is testable against a fake in-memory daemon without a live Nous or the optional
``nous`` extra installed.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from forge.radar.models import Blip, Evidence, Quadrant, Radar, Ring


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()

# --- local JSON backend -----------------------------------------------------

#: Default location for the local JSON store, under the machine-managed ``.forge/`` directory shared
#: with the baton and lessons files.
DEFAULT_JSON_PATH = ".forge/radar/blips.json"


@runtime_checkable
class RadarStore(Protocol):
    """A durable home for the radar. Load returns an empty :class:`Radar` when nothing is stored
    yet, never raising for "not found"."""

    def load(self) -> Radar: ...

    def save(self, radar: Radar) -> None: ...


class JsonRadarStore:
    """The canonical, lossless store: the radar serialised to a JSON file, written atomically
    (tmp + rename) so a crash mid-write never truncates the state."""

    def __init__(self, path: Path | str = DEFAULT_JSON_PATH) -> None:
        self.path = Path(path)

    def load(self) -> Radar:
        if not self.path.is_file():
            return Radar()
        return Radar.model_validate_json(self.path.read_text())

    def save(self, radar: Radar) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(radar.model_dump_json(indent=2))
        tmp.replace(self.path)


# --- Nous backend -----------------------------------------------------------

#: Deterministic namespace so property/row/option ids are stable across saves (uuid5, not uuid4):
#: the same blip keeps the same row id, so a save is an idempotent update rather than a churn.
_NS = uuid.UUID("a11d0000-0000-5000-8000-000000000000")

#: The blip database's logical schema: (column name, Nous type, select options or None), in the
#: order rows should present. Everything is ``text`` for v1: the daemon's ``create_database`` does
#: not honour caller-supplied select option labels (it assigns placeholder "Option" entries), so
#: text columns are the lossless, fully-controlled choice for the machine state store. Quadrant and
#: Ring carry their enum *label*; ``Evidence``/``Links`` carry JSON so the accreted trail survives.
#: (The rendering workstream, which builds the human-facing view, can later swap these to selects.)
RADAR_COLUMNS: list[tuple[str, str, list[str] | None]] = [
    ("Name", "text", None),
    ("Quadrant", "text", None),
    ("Ring", "text", None),
    ("Previous Ring", "text", None),
    ("First Seen", "date", None),
    ("Last Seen", "date", None),
    ("Last Moved", "date", None),
    ("Rationale", "text", None),
    ("Action", "text", None),
    ("Evidence", "text", None),
    ("Links", "text", None),
]

RADAR_DATABASE_TITLE = "Radar Blips"

#: The dedicated notebook the radar lives in — its blip database plus (later) the rendered radar
#: page. Kept apart from the Forge task-tracking notebook so the artifact has its own surface.
RADAR_NOTEBOOK_NAME = "AI Radar"


def _stable_id(*parts: str) -> str:
    return str(uuid.uuid5(_NS, ":".join(parts)))


def radar_database_properties() -> list[dict]:
    """The property definitions for a fresh "Radar Blips" database, with deterministic ids. This is
    what the rendering/provisioning step creates the database from."""
    props: list[dict] = []
    for name, ptype, options in RADAR_COLUMNS:
        prop: dict = {"id": _stable_id("prop", name), "name": name, "type": ptype}
        if options is not None:
            prop["options"] = [
                {"id": _stable_id("option", name, label), "label": label} for label in options
            ]
        props.append(prop)
    return props


def _blip_to_row(blip: Blip, prop_ids: dict[str, str], prev: dict | None, now: str) -> dict:
    """Serialise one blip to a database row. Select cells carry the plain label; ``Evidence`` and
    ``Links`` carry JSON so they round-trip losslessly. Reuses the prior row's id + ``createdAt``
    when the blip is already on the radar (so updates don't churn), and stamps ``updatedAt`` now.

    ``createdAt``/``updatedAt`` are **required** by the frontend's row schema — omitting them makes
    the whole database fail to parse ("cannot parse as v1 or v2")."""
    cells = {
        prop_ids["Name"]: blip.name,
        prop_ids["Quadrant"]: blip.quadrant.value,
        prop_ids["Ring"]: blip.ring.value,
        prop_ids["Previous Ring"]: blip.ring_last.value if blip.ring_last else "",
        prop_ids["First Seen"]: blip.first_seen,
        prop_ids["Last Seen"]: blip.last_seen,
        prop_ids["Last Moved"]: blip.last_moved or "",
        prop_ids["Rationale"]: blip.rationale,
        prop_ids["Action"]: blip.action,
        prop_ids["Evidence"]: json.dumps([e.model_dump() for e in blip.evidence]),
        prop_ids["Links"]: json.dumps(blip.links),
    }
    return {
        "id": (prev or {}).get("id") or _stable_id("row", blip.slug),
        "cells": cells,
        "createdAt": (prev or {}).get("createdAt") or now,
        "updatedAt": now,
    }


def _cell(row: dict, prop_map: dict[str, dict], name: str) -> str:
    """The string value of the *name* column for *row*, or ``""``. Select cells may be stored as an
    option id or a label; both resolve to the label."""
    prop = prop_map.get(name.lower())
    if prop is None:
        return ""
    raw = row.get("cells", {}).get(prop["id"], "")
    if prop.get("type") in ("select", "multiSelect") and raw:
        for opt in prop.get("options", []):
            if opt.get("id") == raw:
                return str(opt.get("label", ""))
    return "" if raw is None else str(raw)


def _row_to_blip(row: dict, prop_map: dict[str, dict]) -> Blip:
    """Deserialise a database row back into a :class:`Blip`. Tolerant of hand edits: an unknown ring
    label or malformed JSON degrades to a sane default rather than raising."""

    def ring_or_none(value: str) -> Ring | None:
        try:
            return Ring(value) if value else None
        except ValueError:
            return None

    def load_json(value: str, default):
        if not value:
            return default
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return default

    evidence = [Evidence.model_validate(e) for e in load_json(_cell(row, prop_map, "Evidence"), [])]
    return Blip(
        name=_cell(row, prop_map, "Name"),
        quadrant=Quadrant(_cell(row, prop_map, "Quadrant")),
        ring=Ring(_cell(row, prop_map, "Ring")),
        ring_last=ring_or_none(_cell(row, prop_map, "Previous Ring")),
        first_seen=_cell(row, prop_map, "First Seen"),
        last_seen=_cell(row, prop_map, "Last Seen"),
        last_moved=_cell(row, prop_map, "Last Moved") or None,
        rationale=_cell(row, prop_map, "Rationale"),
        action=_cell(row, prop_map, "Action"),
        evidence=evidence,
        links=load_json(_cell(row, prop_map, "Links"), []),
    )


@runtime_checkable
class _DatabaseClient(Protocol):
    """The slice of ``nous_mcp.daemon_client.NousDaemonClient`` the Nous store needs. Duck-typed so
    a fake in-memory client can stand in for tests."""

    def get_database(self, notebook_id: str, db_id: str) -> dict: ...

    def put_database(self, notebook_id: str, db_id: str, content: dict) -> dict: ...


@runtime_checkable
class _ProvisionClient(Protocol):
    """The slice of the daemon client needed to find-or-create the radar's notebook and database."""

    def list_notebooks(self) -> list[dict]: ...

    def create_notebook(self, name: str, *, notebook_type: str | None = ...) -> dict: ...

    def list_databases(self, notebook_id: str) -> list[dict]: ...

    def create_database(self, notebook_id: str, title: str, properties: list[dict]) -> dict: ...


def _find_by_name(items: list[dict], name: str, *, key: str) -> dict | None:
    target = name.strip().lower()
    for item in items:
        if str(item.get(key, "")).strip().lower() == target:
            return item
    return None


def provision_radar_store(
    client: _ProvisionClient,
    *,
    notebook_name: str = RADAR_NOTEBOOK_NAME,
    database_title: str = RADAR_DATABASE_TITLE,
    create: bool = True,
) -> NousRadarStore | None:
    """Resolve (and, when ``create``, stand up) the radar's Nous home, returning a ready
    :class:`NousRadarStore`.

    Idempotent: an existing "AI Radar" notebook and "Radar Blips" database are reused as-is; only
    the missing pieces are created (the database from :func:`radar_database_properties`). With
    ``create=False`` this is a pure lookup that returns ``None`` when either piece is absent — the
    read path for ``forge radar status --nous`` before ``init`` has run.
    """
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
        database = client.create_database(notebook_id, database_title, radar_database_properties())

    return NousRadarStore(client, notebook_id, database["id"])


class NousRadarStore:
    """The radar projected into a Nous "Radar Blips" database, one row per blip.

    Round-trips through the daemon's whole-database ``get``/``put``. On save it preserves the
    database's existing property ids and option definitions (only filling gaps deterministically),
    and reuses each blip's existing row id, so repeated saves are idempotent updates not churn.
    """

    def __init__(self, client: _DatabaseClient, notebook_id: str, db_id: str) -> None:
        self.client = client
        self.notebook_id = notebook_id
        self.db_id = db_id

    def _inner(self) -> dict:
        """The daemon nests the actual ``properties``/``rows`` under a ``database`` key (mirroring
        ``nous_mcp.storage.read_database_content``). Fall back to the top level for the fake client
        used in tests, which stores them flat."""
        content = self.client.get_database(self.notebook_id, self.db_id)
        return content.get("database", content)

    def load(self) -> Radar:
        db = self._inner()
        prop_map = {p["name"].lower(): p for p in db.get("properties", [])}
        blips = [_row_to_blip(row, prop_map) for row in db.get("rows", [])]
        return Radar(blips=blips)

    def save(self, radar: Radar) -> None:
        db = self._inner()
        properties = _ensure_properties(db.get("properties", []))
        prop_ids = {p["name"]: p["id"] for p in properties}

        existing = _existing_rows_by_slug(db.get("rows", []), prop_ids.get("Name"))
        now = _now_iso()
        rows = [
            _blip_to_row(blip, prop_ids, existing.get(blip.slug), now) for blip in radar.blips
        ]
        db["properties"] = properties
        db["rows"] = rows
        # Write back the inner database object — the shape
        # ``nous_mcp.storage.write_database_content`` passes to the daemon's whole-database PUT.
        self.client.put_database(self.notebook_id, self.db_id, db)


def _ensure_properties(existing: list[dict]) -> list[dict]:
    """The full radar property list, reusing any existing property's id and merging its select
    options so live edits and daemon-assigned ids survive a save."""
    by_name = {p["name"].lower(): p for p in existing}
    out: list[dict] = []
    for name, ptype, options in RADAR_COLUMNS:
        prior = by_name.get(name.lower())
        prop: dict = {
            "id": prior["id"] if prior else _stable_id("prop", name),
            "name": name,
            "type": ptype,
        }
        if options is not None:
            prior_opts = {o["label"]: o for o in prior.get("options", [])} if prior else {}
            prop["options"] = [
                prior_opts.get(label, {"id": _stable_id("option", name, label), "label": label})
                for label in options
            ]
        out.append(prop)
    return out


def _existing_rows_by_slug(rows: list[dict], name_prop_id: str | None) -> dict[str, dict]:
    """Map ``blip slug -> existing row`` so a save updates rows in place (reusing id + createdAt).
    Keyed by slugifying the Name cell of each existing row."""
    from forge.radar.models import slugify

    if not name_prop_id:
        return {}
    out: dict[str, dict] = {}
    for row in rows:
        name = row.get("cells", {}).get(name_prop_id, "")
        if name and row.get("id"):
            out[slugify(str(name))] = row
    return out
