"""Verify that a finding's cited sources actually exist.

The research model fabricates citations under pressure. Not vague ones — complete, plausible URLs:
a ProPublica article "No New Foreign Payments to Trump Properties in 2025-2026, Audit Finds"
(404), a CREW report URL (404), an `oge.box.com/.../zycb5i2ny8kssm51uzqm8ygyq2zkpkqq.pdf` with a
convincing random hash. The worst of them invented a *negative* finding to support a claim that
something did not happen.

Filtering on "has sources" does not catch this — the sources are non-empty, just fake — so a
fabricated citation propagates into later sprints as prior research and gains authority by
repetition. But existence is decidable without a model: fetch the URL. This is the cheap,
deterministic half of verification, and it runs before anything reaches the knowledge base.

It answers only "does this resolve?" — NOT "does the page support the claim?", which needs judgment
and belongs to the verifier panel (Forge task 790e8d61). A URL that resolves can still be cited for
something it does not say.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

import httpx

# Sources are free text like "Title — Publisher (https://example.com/x)". Only the URL is checkable;
# a citation without one (a case reporter cite, a book) is left alone rather than guessed at.
_URL_RE = re.compile(r"https?://[^\s)\]>,]+")

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Statuses that mean "this document is not there". Everything else — 401/403 paywalls and bot walls,
# 429 rate limits, timeouts, connection errors — is UNKNOWN, not dead. Several real primary sources
# (nycourts.gov, gao.gov, reuters.com) 403 this egress, and discarding them as fabricated would be a
# worse error than keeping a fake: it would quietly delete good research.
_DEAD_STATUSES = frozenset({404, 410})


@dataclass(frozen=True)
class SourceVerdict:
    source: str
    url: str | None
    status: int | None
    dead: bool  # definitively not there (404/410)
    note: str

    @property
    def checkable(self) -> bool:
        return self.url is not None


def extract_url(source: str) -> str | None:
    m = _URL_RE.search(source)
    return m.group(0).rstrip(".,;") if m else None


async def _check_one(client: httpx.AsyncClient, source: str) -> SourceVerdict:
    url = extract_url(source)
    if url is None:
        return SourceVerdict(source, None, None, False, "no url to check")
    try:
        # GET, not HEAD: plenty of sites answer HEAD with 405 or a misleading status.
        resp = await client.get(url)
    except Exception as e:  # network error, timeout, bad host
        return SourceVerdict(source, url, None, False, f"unreachable ({type(e).__name__})")
    dead = resp.status_code in _DEAD_STATUSES
    note = "DEAD" if dead else f"HTTP {resp.status_code}"
    return SourceVerdict(source, url, resp.status_code, dead, note)


async def _check_all(
    sources: list[str], *, timeout: float, proxy: str | None
) -> list[SourceVerdict]:
    limits = httpx.Limits(max_connections=8)
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=timeout,
        headers={"User-Agent": _UA},
        limits=limits,
        proxy=proxy,
    ) as client:
        return list(await asyncio.gather(*(_check_one(client, s) for s in sources)))


def check_sources(
    sources: list[str], *, timeout: float = 15.0, proxy: str | None = None
) -> list[SourceVerdict]:
    """Fetch every cited URL; report which are definitively dead.

    ``proxy`` should match the egress the researcher's own tools use, so a source the researcher
    could reach is one this check can reach too — otherwise blocked-but-real sources look dead from
    here and good citations get dropped.
    """
    if not sources:
        return []
    try:
        return asyncio.run(_check_all(sources, timeout=timeout, proxy=proxy))
    except ImportError as e:
        # A socks5:// proxy needs httpx[socks]. Retry direct rather than killing the run: only
        # 404/410 count as dead, and those are egress-independent in practice, so a direct check
        # still catches fabrications without risking false positives.
        if proxy:
            print(f"    NOTE: proxy unusable for the citation check ({e}); checking directly.")
            return asyncio.run(_check_all(sources, timeout=timeout, proxy=None))
        raise
    except Exception as e:  # noqa: BLE001 — a citation check must never break a research run
        print(f"    NOTE: citation check skipped ({type(e).__name__}: {e})")
        return [SourceVerdict(s, extract_url(s), None, False, "check failed") for s in sources]


def partition_sources(
    sources: list[str], *, timeout: float = 15.0, proxy: str | None = None
) -> tuple[list[str], list[SourceVerdict]]:
    """Split into (kept, dead). Only 404/410 are dropped — see _DEAD_STATUSES on why."""
    verdicts = check_sources(sources, timeout=timeout, proxy=proxy)
    kept = [v.source for v in verdicts if not v.dead]
    dead = [v for v in verdicts if v.dead]
    return kept, dead


def scrub_citations(
    answer: str,
    sources: list[str],
    confidence: str,
    *,
    timeout: float = 15.0,
    proxy: str | None = None,
) -> tuple[str, list[str], str, list[SourceVerdict]]:
    """Drop citations that do not resolve, annotate the answer, and downgrade confidence.

    Returns ``(answer, sources, confidence, dead)``. The annotation is deliberately written into
    the answer text rather than kept as metadata: the verifier panel and any human reader both see
    the finding body, and a fabricated citation is exactly the thing that must not pass quietly.

    Confidence drops to "low" whenever anything was dropped — a finding that cited a source which
    does not exist has demonstrated it will invent one, so its remaining claims deserve suspicion.
    """
    if not sources:
        return answer, sources, confidence, []
    kept, dead = partition_sources(sources, timeout=timeout, proxy=proxy)
    if not dead:
        return answer, kept, confidence, []

    listed = "; ".join(f"{v.url} ({v.note})" for v in dead)
    if kept:
        note = (
            f"\n\n[CITATION CHECK] {len(dead)} of {len(sources)} cited sources do not exist and "
            f"were removed: {listed}. Treat the claims they supported as unsourced."
        )
    else:
        note = (
            f"\n\n[CITATION CHECK] NONE of the {len(sources)} cited sources exist: {listed}. "
            f"Every claim in this finding is unsupported."
        )
    return answer.rstrip() + note, kept, "low", dead
