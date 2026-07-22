"""The candidate feed: entry bookkeeping, JSONL round-trip, and the Nous candidate store."""

from __future__ import annotations

from pathlib import Path

from forge.radar.candidates import (
    CANDIDATE_DATABASE_TITLE,
    CandidateFeed,
    FeedEntry,
    JsonlCandidateStore,
    provision_candidate_store,
)
from forge.radar.models import Quadrant
from forge.radar.sources.base import RawItem


def _item(**kw) -> RawItem:
    base = dict(source="huggingface", external_id="org/model", title="Org Model", url="http://x")
    base.update(kw)
    return RawItem(**base)


def test_feed_entry_from_item_and_refresh_bumps_bookkeeping():
    entry = FeedEntry.from_item(_item(score=10.0), today="2026-07-01")
    assert entry.first_seen == entry.last_seen == "2026-07-01"
    assert entry.times_seen == 1

    later = entry.refreshed(_item(score=20.0, summary="new summary"), today="2026-07-08")
    assert later.first_seen == "2026-07-01"  # preserved
    assert later.last_seen == "2026-07-08"
    assert later.times_seen == 2
    assert later.score == 20.0  # latest score wins
    assert later.summary == "new summary"


def test_feed_upsert_replaces_by_key_in_place():
    feed = CandidateFeed(
        entries=[
            FeedEntry.from_item(_item(external_id="a", title="A"), today="2026-07-01"),
            FeedEntry.from_item(_item(external_id="b", title="B"), today="2026-07-01"),
        ]
    )
    feed.upsert(
        FeedEntry.from_item(_item(external_id="a", title="A", score=99.0), today="2026-07-02")
    )
    assert [e.external_id for e in feed.entries] == ["a", "b"]
    assert feed.by_key()["huggingface:a"].score == 99.0


def _feed() -> CandidateFeed:
    return CandidateFeed(
        entries=[
            FeedEntry.from_item(
                _item(
                    external_id="Qwen/Q",
                    title="Qwen/Q",
                    quadrant_hint=Quadrant.MODELS,
                    score=5.0,
                    published="2026-06-01",
                ),
                today="2026-07-01",
            ),
            FeedEntry.from_item(
                _item(source="hackernews", external_id="42", title="Show HN: a thing"),
                today="2026-07-03",
            ),
        ]
    )


def test_jsonl_store_roundtrips(tmp_path: Path):
    store = JsonlCandidateStore(tmp_path / ".forge" / "radar" / "candidates.jsonl")
    feed = _feed()
    store.save(feed)
    assert store.load().model_dump() == feed.model_dump()


def test_jsonl_store_missing_is_empty(tmp_path: Path):
    assert JsonlCandidateStore(tmp_path / "nope.jsonl").load().entries == []


class FakeProvisionDaemon:
    """Notebooks + databases fake, shared shape with the blip-store tests."""

    def __init__(self) -> None:
        self.notebooks: list[dict] = []
        self.databases: dict[str, list[dict]] = {}
        self.contents: dict[str, dict] = {}
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


def test_provision_candidate_store_creates_in_named_notebook():
    daemon = FakeProvisionDaemon()
    store = provision_candidate_store(daemon, notebook_name="AI Radar")
    assert store is not None
    assert [nb["name"] for nb in daemon.notebooks] == ["AI Radar"]
    assert [db["title"] for db in daemon.databases[store.notebook_id]] == [CANDIDATE_DATABASE_TITLE]


def test_provision_candidate_create_false_returns_none_when_absent():
    assert (
        provision_candidate_store(FakeProvisionDaemon(), notebook_name="AI Radar", create=False)
        is None
    )


def test_nous_candidate_store_roundtrips_and_reuses_row_ids():
    daemon = FakeProvisionDaemon()
    store = provision_candidate_store(daemon, notebook_name="AI Radar")
    feed = _feed()
    store.save(feed)
    assert store.load().model_dump() == feed.model_dump()

    ids1 = {r["id"] for r in daemon.contents[store.db_id]["rows"]}
    store.save(store.load())
    ids2 = {r["id"] for r in daemon.contents[store.db_id]["rows"]}
    assert ids1 == ids2 and len(ids2) == 2  # idempotent, no duplicate rows


def test_nous_candidate_store_reuses_existing_notebook_and_db():
    daemon = FakeProvisionDaemon()
    provision_candidate_store(daemon, notebook_name="AI Radar")
    provision_candidate_store(daemon, notebook_name="AI Radar")
    assert len(daemon.notebooks) == 1
    assert len(daemon.databases[daemon.notebooks[0]["id"]]) == 1
