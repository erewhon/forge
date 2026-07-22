"""arXiv recent submissions in cs.CL / cs.AI / cs.LG → the Techniques quadrant.

API (no auth): ``GET http://export.arxiv.org/api/query?search_query=cat:cs.CL+OR+cat:cs.AI+OR+
cat:cs.LG&sortBy=submittedDate&sortOrder=descending&max_results=N``. The response is **Atom XML**
(not JSON), parsed here with the stdlib ``xml.etree``. Per ``<entry>`` (live-captured 2026-07):

- ``<id>`` — ``http://arxiv.org/abs/2501.01234v1``; the abs id (minus version) is the external id.
- ``<title>`` / ``<summary>`` — whitespace-wrapped in the feed, so both are re-flattened.
- ``<published>`` — ISO timestamp.

arXiv has no popularity signal, so ``score`` is ``None``. The hint is Techniques (methods/how-to-
build), though many entries are really about Models — the classifier refines from the abstract.
"""

from __future__ import annotations

import re
from xml.etree import ElementTree as ET

import httpx

from forge.radar.models import Quadrant
from forge.radar.sources.base import RawItem

API_URL = "http://export.arxiv.org/api/query"
DEFAULT_QUERY = "cat:cs.CL OR cat:cs.AI OR cat:cs.LG"

_ATOM = "{http://www.w3.org/2005/Atom}"
_ABS_ID_RE = re.compile(r"arxiv\.org/abs/(?P<id>.+?)(?:v\d+)?$")


def _flatten(text: str | None) -> str:
    """Collapse the feed's line-wrapped whitespace into a single spaced string."""
    return re.sub(r"\s+", " ", (text or "").strip())


class ArxivAdapter:
    name = "arxiv"

    def __init__(self, query: str = DEFAULT_QUERY, limit: int = 40) -> None:
        self.query = query
        self.limit = limit

    def fetch(self, client: httpx.Client) -> list[RawItem]:
        resp = client.get(
            API_URL,
            params={
                "search_query": self.query,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
                "max_results": str(self.limit),
            },
        )
        resp.raise_for_status()
        return self.parse(resp.text)

    def parse(self, xml_text: str) -> list[RawItem]:
        root = ET.fromstring(xml_text)
        items: list[RawItem] = []
        for entry in root.findall(f"{_ATOM}entry"):
            raw_id = _flatten(entry.findtext(f"{_ATOM}id"))
            match = _ABS_ID_RE.search(raw_id)
            external_id = match.group("id") if match else raw_id
            title = _flatten(entry.findtext(f"{_ATOM}title"))
            if not external_id or not title:
                continue
            items.append(
                RawItem(
                    source=self.name,
                    external_id=external_id,
                    title=title,
                    url=raw_id or f"https://arxiv.org/abs/{external_id}",
                    summary=_flatten(entry.findtext(f"{_ATOM}summary"))[:500],
                    quadrant_hint=Quadrant.TECHNIQUES,
                    score=None,
                    published=entry.findtext(f"{_ATOM}published"),
                )
            )
        return items
