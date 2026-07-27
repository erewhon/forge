"""Citation existence checking — the deterministic half of verification.

The research model fabricates citations: a ProPublica article "No New Foreign Payments to Trump
Properties in 2025-2026, Audit Finds" (404), a CREW report URL (404), an oge.box.com PDF with a
convincing random hash. Filtering on "has sources" misses all of it, because the sources are
non-empty — just fake. Existence is decidable without a model, so it is checked before a finding is
written.

The critical safety property is the OTHER direction: a 403 or a timeout must NEVER be treated as
fabrication. Real primary sources (nycourts.gov, gao.gov, reuters.com) block this egress, and
deleting them as invented would silently destroy good research.
"""

from __future__ import annotations

import httpx
import pytest

from forge.shared import source_check
from forge.shared.source_check import extract_url, scrub_citations

REAL = "Codifying the Emoluments Clauses — Brennan Center (https://www.brennancenter.org/x.pdf)"
FAKE = (
    "No New Foreign Payments to Trump Properties in 2025-2026, Audit Finds — ProPublica "
    "(https://www.propublica.org/article/trump-properties-foreign-payments-2025-2026)"
)
PAYWALLED = "Trump's Hotels Took Foreign Cash — Reuters (https://www.reuters.com/x)"
NO_URL = "CREW v. Trump, 941 F.3d 607 (2d Cir. 2019) — Second Circuit reverses dismissal"


def _fake_transport(status_by_host: dict[str, int]):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_by_host.get(request.url.host, 200))

    return httpx.MockTransport(handler)


@pytest.fixture
def patched(monkeypatch):
    """Route the checker's client through a MockTransport — no network in tests."""
    statuses = {
        "www.propublica.org": 404,
        "crew.org": 404,
        "www.reuters.com": 403,
        "www.brennancenter.org": 200,
    }
    real_client = httpx.AsyncClient

    def client_factory(**kwargs):
        kwargs.pop("proxy", None)
        return real_client(transport=_fake_transport(statuses), **kwargs)

    monkeypatch.setattr(source_check.httpx, "AsyncClient", client_factory)


def test_extract_url_handles_trailing_punctuation():
    assert extract_url("Title — Pub (https://example.com/a)") == "https://example.com/a"
    assert extract_url("see https://example.com/b, later") == "https://example.com/b"
    assert extract_url(NO_URL) is None


def test_fabricated_citation_is_dropped_and_flagged(patched):
    answer, sources, conf, dead = scrub_citations("Findings text.", [REAL, FAKE], "high")
    assert sources == [REAL]
    assert len(dead) == 1
    assert conf == "low", "a finding that invented a citation must not stay high-confidence"
    assert "[CITATION CHECK]" in answer
    assert "propublica" in answer


def test_paywalled_source_is_kept(patched):
    """403 is a bot wall, not proof of fabrication — dropping it would delete real research."""
    answer, sources, conf, dead = scrub_citations("Findings.", [PAYWALLED], "medium")
    assert sources == [PAYWALLED]
    assert dead == []
    assert conf == "medium"
    assert "[CITATION CHECK]" not in answer


def test_citation_without_url_is_left_alone(patched):
    answer, sources, conf, dead = scrub_citations("Findings.", [NO_URL], "high")
    assert sources == [NO_URL]
    assert dead == []
    assert conf == "high"


def test_all_citations_fabricated_says_so(patched):
    answer, sources, conf, dead = scrub_citations("Sweeping claim.", [FAKE], "high")
    assert sources == []
    assert conf == "low"
    assert "NONE of the" in answer
    assert "Every claim in this finding is unsupported" in answer


def test_no_sources_is_a_noop(patched):
    answer, sources, conf, dead = scrub_citations("Text.", [], "low")
    assert (answer, sources, conf, dead) == ("Text.", [], "low", [])


def test_unreachable_host_is_not_dead(monkeypatch):
    """A connection error is unknown, not fabricated."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    real_client = httpx.AsyncClient

    def client_factory(**kwargs):
        kwargs.pop("proxy", None)
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(source_check.httpx, "AsyncClient", client_factory)
    _, sources, conf, dead = scrub_citations("Text.", [REAL], "high")
    assert sources == [REAL]
    assert dead == []
    assert conf == "high"


def test_socks_proxy_without_socksio_falls_back_to_direct(monkeypatch, capsys):
    """A missing httpx[socks] must degrade to a direct check, not kill the research run."""
    calls: list[str | None] = []
    real_client = httpx.AsyncClient

    def client_factory(**kwargs):
        proxy = kwargs.pop("proxy", None)
        calls.append(proxy)
        if proxy:
            raise ImportError("Using SOCKS proxy, but the 'socksio' package is not installed.")
        return real_client(transport=_fake_transport({"www.propublica.org": 404}), **kwargs)

    monkeypatch.setattr(source_check.httpx, "AsyncClient", client_factory)
    _, sources, _, dead = scrub_citations(
        "Text.", [FAKE], "high", proxy="socks5://192.168.42.84:1080"
    )
    assert calls == ["socks5://192.168.42.84:1080", None], "must retry without the proxy"
    assert len(dead) == 1, "the direct retry still catches the fabrication"
    assert sources == []


def test_unexpected_failure_keeps_every_source(monkeypatch):
    """If the check itself breaks, keep the research — never drop sources on a tooling error."""

    def boom(**kwargs):
        raise RuntimeError("event loop exploded")

    monkeypatch.setattr(source_check.httpx, "AsyncClient", boom)
    answer, sources, conf, dead = scrub_citations("Text.", [REAL, FAKE], "high")
    assert sources == [REAL, FAKE]
    assert dead == []
    assert conf == "high"
    assert "[CITATION CHECK]" not in answer
