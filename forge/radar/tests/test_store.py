"""Store backends: the JSON store's lossless round-trip, and the Nous store's whole-database
round-trip + idempotent ids, exercised against a fake in-memory daemon (no live Nous)."""

from __future__ import annotations

from pathlib import Path

from forge.radar.models import Blip, Evidence, Quadrant, Radar, Ring
from forge.radar.store import (
    RADAR_DATABASE_TITLE,
    RADAR_NOTEBOOK_NAME,
    JsonRadarStore,
    NousRadarStore,
    provision_radar_store,
    radar_database_properties,
)


def _radar() -> Radar:
    return Radar(
        blips=[
            Blip(
                name="Qwen3-Coder 30B",
                quadrant=Quadrant.MODELS,
                ring=Ring.TRIAL,
                ring_last=Ring.ASSESS,
                first_seen="2026-06-01",
                last_seen="2026-07-20",
                last_moved="2026-07-15",
                rationale="strong on the euclid router",
                action="trial as the default coder tier",
                evidence=[
                    Evidence(date="2026-06-01", note="First seen", source="hf"),
                    Evidence(
                        date="2026-07-15", note="Assess → Trial: benchmarked", source="hands-on"
                    ),
                ],
                links=["https://hf.co/qwen"],
            ),
            Blip(
                name="Structured tool-calling",
                quadrant=Quadrant.TECHNIQUES,
                ring=Ring.ASSESS,
                first_seen="2026-07-10",
                last_seen="2026-07-10",
            ),
        ]
    )


# --- JSON store --------------------------------------------------------------


def test_json_store_roundtrips_losslessly(tmp_path: Path):
    store = JsonRadarStore(tmp_path / ".forge" / "radar" / "blips.json")
    radar = _radar()
    store.save(radar)
    loaded = store.load()
    assert loaded.model_dump() == radar.model_dump()


def test_json_store_load_missing_returns_empty(tmp_path: Path):
    store = JsonRadarStore(tmp_path / "nope.json")
    assert store.load().blips == []


def test_json_store_save_is_atomic_no_tmp_left(tmp_path: Path):
    path = tmp_path / "blips.json"
    JsonRadarStore(path).save(_radar())
    assert path.exists()
    assert not path.with_suffix(".json.tmp").exists()


# --- Nous store --------------------------------------------------------------


class FakeDaemon:
    """A minimal in-memory stand-in for the slice of NousDaemonClient the store uses. Holds one
    database's content dict, seeded (optionally) with pre-existing properties/rows."""

    def __init__(self, content: dict | None = None) -> None:
        self.content = content or {"properties": [], "rows": []}
        self.put_calls = 0

    def get_database(self, notebook_id: str, db_id: str) -> dict:
        # Return a copy so the store can't mutate our stored content except via put.
        import copy

        return copy.deepcopy(self.content)

    def put_database(self, notebook_id: str, db_id: str, content: dict) -> dict:
        self.content = content
        self.put_calls += 1
        return {"ok": True}


def test_nous_store_roundtrips_through_a_fresh_database():
    daemon = FakeDaemon()  # empty DB, as if just created
    store = NousRadarStore(daemon, notebook_id="nb", db_id="db")
    radar = _radar()

    store.save(radar)
    loaded = store.load()

    assert loaded.model_dump() == radar.model_dump()


def test_nous_store_rows_carry_required_timestamps():
    # The frontend row schema requires createdAt/updatedAt — omitting them makes the whole database
    # fail to parse ("cannot parse as v1 or v2").
    daemon = FakeDaemon()
    store = NousRadarStore(daemon, notebook_id="nb", db_id="db")
    store.save(_radar())
    rows = daemon.content["rows"]
    assert rows and all(r.get("createdAt") and r.get("updatedAt") for r in rows)

    # A second save preserves createdAt and refreshes updatedAt.
    created_before = {r["id"]: r["createdAt"] for r in daemon.content["rows"]}
    store.save(store.load())
    for r in daemon.content["rows"]:
        assert r["createdAt"] == created_before[r["id"]]


def test_nous_store_reuses_row_ids_across_saves():
    daemon = FakeDaemon()
    store = NousRadarStore(daemon, notebook_id="nb", db_id="db")

    store.save(_radar())
    first_ids = {r["id"] for r in daemon.content["rows"]}

    # A second save of the same blips must update the same rows, not create new ones.
    store.save(store.load())
    second_ids = {r["id"] for r in daemon.content["rows"]}

    assert first_ids == second_ids
    assert len(daemon.content["rows"]) == 2


def test_nous_store_preserves_existing_property_ids():
    # Simulate a DB whose "Ring" property already has a daemon-assigned id.
    daemon = FakeDaemon(
        {
            "properties": [
                {"id": "existing-ring-id", "name": "Ring", "type": "text"},
            ],
            "rows": [],
        }
    )
    store = NousRadarStore(daemon, notebook_id="nb", db_id="db")
    store.save(_radar())

    ring_prop = next(p for p in daemon.content["properties"] if p["name"] == "Ring")
    assert ring_prop["id"] == "existing-ring-id"  # id preserved, not regenerated


def test_radar_database_properties_schema_shape():
    props = radar_database_properties()
    names = [p["name"] for p in props]
    assert names[0] == "Name"
    assert "Quadrant" in names and "Ring" in names
    # v1 is all-text columns (the daemon mangles caller-supplied select options), so nothing
    # carries an options list.
    assert all(p["type"] == "text" or p["type"] == "date" for p in props)
    assert not any("options" in p for p in props)


class FakeProvisionDaemon:
    """A fuller fake with notebooks + databases, for the provisioning path. Also supports
    get/put_database so a provisioned store round-trips."""

    def __init__(self) -> None:
        self.notebooks: list[dict] = []
        self.databases: dict[str, list[dict]] = {}  # notebook_id -> [db, ...]
        self.contents: dict[str, dict] = {}  # db_id -> content
        self._n = 0

    def _id(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}-{self._n}"

    def list_notebooks(self) -> list[dict]:
        return list(self.notebooks)

    def create_notebook(self, name: str, *, notebook_type: str | None = None) -> dict:
        nb = {"id": self._id("nb"), "name": name}
        self.notebooks.append(nb)
        self.databases[nb["id"]] = []
        return nb

    def list_databases(self, notebook_id: str) -> list[dict]:
        return list(self.databases.get(notebook_id, []))

    def create_database(self, notebook_id: str, title: str, properties: list[dict]) -> dict:
        db = {"id": self._id("db"), "title": title}
        self.databases.setdefault(notebook_id, []).append(db)
        self.contents[db["id"]] = {"properties": properties, "rows": []}
        return db

    def get_database(self, notebook_id: str, db_id: str) -> dict:
        import copy

        return copy.deepcopy(self.contents[db_id])

    def put_database(self, notebook_id: str, db_id: str, content: dict) -> dict:
        self.contents[db_id] = content
        return {"ok": True}


def test_provision_creates_notebook_and_database_when_absent():
    daemon = FakeProvisionDaemon()
    store = provision_radar_store(daemon)
    assert store is not None
    assert [nb["name"] for nb in daemon.notebooks] == [RADAR_NOTEBOOK_NAME]
    assert [db["title"] for db in daemon.databases[store.notebook_id]] == [RADAR_DATABASE_TITLE]


def test_provision_is_idempotent():
    daemon = FakeProvisionDaemon()
    first = provision_radar_store(daemon)
    second = provision_radar_store(daemon)
    assert (first.notebook_id, first.db_id) == (second.notebook_id, second.db_id)
    assert len(daemon.notebooks) == 1  # no duplicate notebook
    assert len(daemon.databases[first.notebook_id]) == 1  # no duplicate database


def test_provision_create_false_returns_none_when_absent():
    daemon = FakeProvisionDaemon()
    assert provision_radar_store(daemon, create=False) is None


def test_provisioned_store_roundtrips_a_radar():
    daemon = FakeProvisionDaemon()
    store = provision_radar_store(daemon)
    store.save(_radar())
    assert provision_radar_store(daemon, create=False).load().model_dump() == _radar().model_dump()


def test_nous_store_tolerates_hand_edited_bad_ring_and_json():
    # A row with an unknown ring label and malformed Evidence JSON must degrade, not crash.
    props = radar_database_properties()
    pid = {p["name"]: p["id"] for p in props}
    daemon = FakeDaemon(
        {
            "properties": props,
            "rows": [
                {
                    "id": "r1",
                    "cells": {
                        pid["Name"]: "Weird",
                        pid["Quadrant"]: "Techniques",
                        pid["Ring"]: "Assess",
                        pid["Previous Ring"]: "Bogus",  # not a real ring → None
                        pid["First Seen"]: "2026-07-01",
                        pid["Last Seen"]: "2026-07-01",
                        pid["Evidence"]: "{not json",  # malformed → []
                        pid["Links"]: "",
                    },
                }
            ],
        }
    )
    store = NousRadarStore(daemon, notebook_id="nb", db_id="db")
    radar = store.load()
    blip = radar.get("Weird")
    assert blip.ring_last is None
    assert blip.evidence == []
