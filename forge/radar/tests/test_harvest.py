"""Harvest orchestration: relevance drop, blip skip, within-source dedup, and a dark source that
does not abort the run."""

from __future__ import annotations

from datetime import date

from forge.radar.candidates import CandidateFeed, FeedEntry
from forge.radar.harvest import harvest
from forge.radar.models import Quadrant
from forge.radar.sources.base import RawItem

D = date(2026, 7, 10)


class StubAdapter:
    """An adapter that returns fixed items (or raises) — no network."""

    def __init__(self, name: str, items: list[RawItem] | None = None, exc: Exception | None = None):
        self.name = name
        self._items = items or []
        self._exc = exc

    def fetch(self, client) -> list[RawItem]:
        if self._exc is not None:
            raise self._exc
        return self._items


def _item(name: str, source: str = "hackernews", ext: str = "1", **kw) -> RawItem:
    return RawItem(source=source, external_id=ext, title=name, url="http://x", **kw)


def test_relevant_items_are_added_and_irrelevant_dropped():
    feed = CandidateFeed()
    adapter = StubAdapter(
        "hackernews",
        [
            _item("A new LLM agent framework", ext="1"),
            _item("Best ergonomic chair for developers", ext="2"),  # not AI → dropped
        ],
    )
    report = harvest(feed, [adapter], client=None, blip_slugs=set(), today=D)

    sr = report.sources[0]
    assert sr.fetched == 2 and sr.added == 1 and sr.dropped_irrelevant == 1
    assert [e.title for e in feed.entries] == ["A new LLM agent framework"]
    assert feed.entries[0].first_seen == "2026-07-10"


def test_items_already_blips_are_skipped():
    feed = CandidateFeed()
    adapter = StubAdapter(
        "huggingface",
        [
            _item(
                "Qwen3-Coder",
                source="huggingface",
                ext="Qwen/Qwen3-Coder",
                quadrant_hint=Quadrant.MODELS,
            ),
        ],
    )
    report = harvest(feed, [adapter], client=None, blip_slugs={"qwen3-coder"}, today=D)
    assert report.sources[0].skipped_blip == 1
    assert feed.entries == []


def test_reseen_item_refreshes_instead_of_duplicating():
    feed = CandidateFeed(
        entries=[
            FeedEntry.from_item(_item("An LLM agent", ext="1", score=10.0), today="2026-07-01"),
        ]
    )
    adapter = StubAdapter("hackernews", [_item("An LLM agent", ext="1", score=25.0)])
    report = harvest(feed, [adapter], client=None, blip_slugs=set(), today=D)

    assert report.sources[0].refreshed == 1 and report.sources[0].added == 0
    assert len(feed.entries) == 1
    entry = feed.entries[0]
    assert entry.times_seen == 2 and entry.last_seen == "2026-07-10" and entry.score == 25.0
    assert entry.first_seen == "2026-07-01"


def test_missing_quadrant_hint_is_filled_by_the_classifier():
    feed = CandidateFeed()
    adapter = StubAdapter(
        "hackernews", [_item("A vLLM inference serving benchmark on GPU", ext="9")]
    )
    harvest(feed, [adapter], client=None, blip_slugs=set(), today=D)
    assert feed.entries[0].quadrant_hint is Quadrant.INFRA


def test_dark_source_is_recorded_not_fatal():
    feed = CandidateFeed()
    good = StubAdapter("huggingface", [_item("An LLM model", source="huggingface", ext="m")])
    bad = StubAdapter("github", exc=RuntimeError("503 upstream"))
    report = harvest(feed, [bad, good], client=None, blip_slugs=set(), today=D)

    by_name = {s.source: s for s in report.sources}
    assert "503 upstream" in by_name["github"].error
    assert by_name["huggingface"].added == 1  # the good source still ran
    assert report.added == 1


def test_report_render_lists_sources_and_total():
    feed = CandidateFeed()
    adapter = StubAdapter("hackernews", [_item("An LLM agent", ext="1")])
    report = harvest(feed, [adapter], client=None, blip_slugs=set(), today=D)
    text = report.render()
    assert "hackernews" in text
    assert "1 new candidate" in text
