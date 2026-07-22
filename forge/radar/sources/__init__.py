"""Source adapters — the noisy, frequent layer that accumulates candidate signals for the radar.

Each adapter hits one free/clean source and emits normalized
:class:`~forge.radar.sources.base.RawItem` records; the :mod:`forge.radar.harvest` orchestrator
relevance-filters them, dedups against the feed and the blip store, and appends survivors to the
candidate feed. Adapters do **no** judgment beyond a per-source ``quadrant_hint`` — relevance and
placement are the weekly synthesis's job.

Adapters are split into a pure ``parse(payload)`` (tested against captured fixtures) and a thin
``fetch(client)`` that does the HTTP call and hands the payload to ``parse``, so the parsing logic
is covered without network access.
"""

from __future__ import annotations

from forge.radar.sources.arxiv import ArxivAdapter
from forge.radar.sources.base import RawItem, SourceAdapter, radar_http_client
from forge.radar.sources.github import GitHubAdapter
from forge.radar.sources.hackernews import HackerNewsAdapter
from forge.radar.sources.huggingface import HuggingFaceAdapter


def default_adapters() -> list[SourceAdapter]:
    """The adapters run by ``forge radar scan`` with no source filter. Clean JSON/Atom APIs first;
    Reddit and provider changelogs are deferred follow-ups."""
    return [
        HuggingFaceAdapter(),
        GitHubAdapter(),
        ArxivAdapter(),
        HackerNewsAdapter(),
    ]


__all__ = [
    "ArxivAdapter",
    "GitHubAdapter",
    "HackerNewsAdapter",
    "HuggingFaceAdapter",
    "RawItem",
    "SourceAdapter",
    "default_adapters",
    "radar_http_client",
]
