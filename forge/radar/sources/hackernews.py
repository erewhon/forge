"""Hacker News front page via the Algolia API → unclassified (the classifier reads the title).

API (no auth): ``GET https://hn.algolia.com/api/v1/search?tags=front_page&hitsPerPage=N``. The front
page is a genuine firehose — mostly *not* about AI — which is exactly why it exercises the relevance
gate: the harvester drops everything the gate rejects, keeping only the AI-shaped stories. Fields
used (live-captured 2026-07):

- ``objectID`` — the external id; also the HN item id for the comments URL.
- ``title`` — the headline (a headline, *not* an entity, so these dedup poorly and are the ones the
  synthesis will most need to consolidate/rename).
- ``url`` — the linked article (may be absent for text posts → fall back to the HN item page).
- ``points`` — the radar score.
- ``created_at`` — ISO timestamp.

No ``quadrant_hint`` — a headline could be any quadrant, so the classifier decides from the title.
"""

from __future__ import annotations

import httpx

from forge.radar.sources.base import RawItem

API_URL = "https://hn.algolia.com/api/v1/search"


class HackerNewsAdapter:
    name = "hackernews"

    def __init__(self, tags: str = "front_page", limit: int = 50) -> None:
        self.tags = tags
        self.limit = limit

    def fetch(self, client: httpx.Client) -> list[RawItem]:
        resp = client.get(
            API_URL,
            params={"tags": self.tags, "hitsPerPage": str(self.limit)},
        )
        resp.raise_for_status()
        return self.parse(resp.json())

    def parse(self, payload: dict) -> list[RawItem]:
        items: list[RawItem] = []
        for hit in payload.get("hits", []):
            object_id = hit.get("objectID")
            title = hit.get("title") or hit.get("story_title")
            if not object_id or not title:
                continue
            points = hit.get("points")
            items.append(
                RawItem(
                    source=self.name,
                    external_id=str(object_id),
                    title=title,
                    url=hit.get("url") or f"https://news.ycombinator.com/item?id={object_id}",
                    summary="",  # HN stories carry no abstract; the title is all the gate has.
                    quadrant_hint=None,
                    score=float(points) if isinstance(points, (int, float)) else None,
                    published=hit.get("created_at"),
                )
            )
        return items
