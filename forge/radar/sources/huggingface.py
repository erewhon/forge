"""HuggingFace trending models → the Models quadrant.

API (no auth): ``GET https://huggingface.co/api/models?sort=trendingScore&direction=-1&limit=N``.
Returns a JSON array of model records; the fields used here (live-captured 2026-07):

- ``id`` — ``"org/name"``, the stable external id and the model's page slug.
- ``likes`` — the popularity score used as the radar signal.
- ``pipeline_tag`` — e.g. ``"text-generation"``; joined with a few ``tags`` as the summary.
- ``createdAt`` / ``lastModified`` — ISO timestamps (present with ``full`` records).

Everything on HF is a model, so the relevance gate passes nearly all of these — the point of the
adapter is breadth of the Models quadrant, not filtering.
"""

from __future__ import annotations

import httpx

from forge.radar.models import Quadrant
from forge.radar.sources.base import RawItem

API_URL = "https://huggingface.co/api/models"


class HuggingFaceAdapter:
    name = "huggingface"

    def __init__(self, limit: int = 40, sort: str = "trendingScore") -> None:
        self.limit = limit
        self.sort = sort

    def fetch(self, client: httpx.Client) -> list[RawItem]:
        resp = client.get(
            API_URL,
            params={"sort": self.sort, "direction": "-1", "limit": str(self.limit)},
        )
        resp.raise_for_status()
        return self.parse(resp.json())

    def parse(self, payload: list[dict]) -> list[RawItem]:
        items: list[RawItem] = []
        for m in payload:
            model_id = m.get("id") or m.get("modelId")
            if not model_id:
                continue
            tags = [t for t in (m.get("tags") or []) if isinstance(t, str)]
            summary_bits = [m.get("pipeline_tag")] + tags[:6]
            summary = ", ".join(b for b in summary_bits if b)
            likes = m.get("likes")
            items.append(
                RawItem(
                    source=self.name,
                    external_id=model_id,
                    title=model_id,
                    url=f"https://huggingface.co/{model_id}",
                    summary=summary,
                    quadrant_hint=Quadrant.MODELS,
                    score=float(likes) if isinstance(likes, (int, float)) else None,
                    published=m.get("createdAt") or m.get("lastModified"),
                )
            )
        return items
