"""The adapter contract: :class:`RawItem` (one normalized source signal) and :class:`SourceAdapter`
(the protocol every source implements), plus a shared HTTP client.

Adapters split into two halves so parsing is testable without a network:

- ``parse(payload) -> list[RawItem]`` — pure; the tests feed it a captured fixture.
- ``fetch(client) -> list[RawItem]`` — does the HTTP request(s) and calls ``parse``.

A :class:`RawItem` is source-native and judgment-free: it is *not* a blip. It carries a stable
``external_id`` (for exact within-source dedup), the human ``title`` (slugified for dedup against
the blip store), a ``quadrant_hint`` (the adapter's weak prior), and whatever popularity ``score``
the source exposes. Turning items into blips is the synthesis's job.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import httpx
from pydantic import BaseModel

from forge.radar.models import Quadrant

#: A courteous, identifiable UA — several of these APIs ask for one and rate-limit anonymous bulk
#: traffic harder.
USER_AGENT = "forge-radar/0.1 (+https://code.middlefork.org; AI tech radar scanner)"

_TIMEOUT = 20.0


def radar_http_client() -> httpx.Client:
    """A shared HTTP client for the adapters: identified UA, sane timeout, redirects followed."""
    return httpx.Client(
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        timeout=_TIMEOUT,
        follow_redirects=True,
    )


class RawItem(BaseModel):
    """One normalized signal from a source. Source-native, not a blip."""

    source: str  #: Adapter name, e.g. "huggingface". Half of the dedup key.
    external_id: str  #: Stable id *within* the source (model id, arXiv id, HN objectID, repo). Half
    #: of the dedup key.
    title: str  #: Human name/headline. Slugified for dedup against the blip store.
    url: str
    summary: str = ""  #: Abstract / description / tags — the text the relevance filter reads.
    quadrant_hint: Quadrant | None = None  #: The adapter's weak prior; synthesis decides for real.
    score: float | None = (
        None  #: Popularity signal (likes / stars / points), if the source has one.
    )
    published: str | None = None  #: ISO-8601 date the item was published, if known.

    @property
    def key(self) -> str:
        """The exact within-source dedup key."""
        return f"{self.source}:{self.external_id}"


@runtime_checkable
class SourceAdapter(Protocol):
    """A radar source. ``name`` is the ``RawItem.source`` stamp and the ``--source`` selector."""

    name: str

    def fetch(self, client: httpx.Client) -> list[RawItem]:
        """Hit the source and return normalized items. Network failures are the caller's to handle;
        an adapter should let them propagate rather than swallow a source going dark."""
        ...
