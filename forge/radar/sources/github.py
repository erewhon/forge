"""GitHub search for active AI repos → the Agents & Frameworks / Infra quadrants.

There is no official "trending" API, so this uses the Search API (no auth, 10 req/min for anonymous
callers — fine for a weekly scan): ``GET https://api.github.com/search/repositories?q=...``. The
default query targets agent/LLM/MCP tooling pushed recently, most-starred first. Fields used
(live-captured 2026-07):

- ``full_name`` — ``"org/repo"``, the external id and display name.
- ``html_url`` / ``description`` / ``stargazers_count`` — url, summary, and the radar score.
- ``pushed_at`` — recency (used as ``published``).

The adapter leaves ``quadrant_hint`` unset: GitHub repos split between Agents & Frameworks and
Infra/Tooling, so the classifier decides from the description rather than a blanket prior.
"""

from __future__ import annotations

import httpx

from forge.radar.sources.base import RawItem

API_URL = "https://api.github.com/search/repositories"

#: Default search: agent/LLM/MCP tooling, most-starred. ``sort``/``order`` are separate params.
DEFAULT_QUERY = "agent OR llm OR mcp OR agentic in:name,description stars:>200"


class GitHubAdapter:
    name = "github"

    def __init__(self, query: str = DEFAULT_QUERY, limit: int = 30) -> None:
        self.query = query
        self.limit = limit

    def fetch(self, client: httpx.Client) -> list[RawItem]:
        resp = client.get(
            API_URL,
            params={
                "q": self.query,
                "sort": "stars",
                "order": "desc",
                "per_page": str(self.limit),
            },
            headers={"Accept": "application/vnd.github+json"},
        )
        resp.raise_for_status()
        return self.parse(resp.json())

    def parse(self, payload: dict) -> list[RawItem]:
        items: list[RawItem] = []
        for r in payload.get("items", []):
            full_name = r.get("full_name")
            if not full_name:
                continue
            stars = r.get("stargazers_count")
            items.append(
                RawItem(
                    source=self.name,
                    external_id=full_name,
                    title=full_name,
                    url=r.get("html_url") or f"https://github.com/{full_name}",
                    summary=r.get("description") or "",
                    quadrant_hint=None,  # Agents vs Infra — classifier reads the description.
                    score=float(stars) if isinstance(stars, (int, float)) else None,
                    published=r.get("pushed_at"),
                )
            )
        return items
