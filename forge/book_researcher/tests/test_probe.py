"""`forge book probe`: host extraction, verdicts, persistence — with an injected fetcher."""

from __future__ import annotations

from forge.book_researcher import main as main_mod
from forge.book_researcher.models import BookConfig, ChapterOutline, SourcePolicy
from forge.book_researcher.probe import (
    classify_status,
    hosts_in_config,
    load_reachability,
    probe_book,
    probe_hosts,
    render_reachability,
)


def _book() -> BookConfig:
    return BookConfig(
        title="T",
        description="D",
        sources=SourcePolicy(reachable=["https://courtlistener.com/"], blocked=["nycourts.gov"]),
        chapters=[
            ChapterOutline(
                number=1,
                title="One",
                description="Uses EDGAR full-text search at efts.sec.gov.",
                sources=["fec.gov/data"],
                research_questions=[
                    "What does usaspending.gov show, e.g. award IDs, vs. what GAO said?",
                    "Per https://www.sec.gov/cgi-bin/browse-edgar?x=1, what was filed?",
                ],
            )
        ],
    )


def test_hosts_in_config_finds_hosts_everywhere_and_ignores_abbreviations():
    hosts = hosts_in_config(_book())
    assert hosts == {
        "courtlistener.com",
        "nycourts.gov",
        "efts.sec.gov",
        "fec.gov",
        "usaspending.gov",
        "www.sec.gov",
    }


def test_classify_status():
    assert classify_status(200) == ("reachable", "HTTP 200")
    assert classify_status(404)[0] == "reachable"  # root 404 still means the host answers
    assert classify_status(403)[0] == "blocked"
    assert classify_status(401)[0] == "blocked"
    assert classify_status(503)[0] == "unknown"
    assert classify_status(None, "ConnectTimeout") == ("unknown", "unreachable (ConnectTimeout)")


def _fake_fetch(table: dict[str, tuple[int | None, str | None]]):
    def fetch_many(urls: list[str]):
        return [table.get(u, (None, "ConnectError")) for u in urls]

    return fetch_many


def test_probe_hosts_is_sorted_and_classified():
    verdicts = probe_hosts(
        {"b.gov", "a.gov"},
        fetch_many=_fake_fetch({"https://a.gov/": (200, None), "https://b.gov/": (403, None)}),
    )
    assert list(verdicts) == ["a.gov", "b.gov"]
    assert verdicts["a.gov"].state == "reachable"
    assert verdicts["b.gov"].state == "blocked"


def test_probe_book_persists_and_merges_with_previous(tmp_path):
    book = _book()
    first = probe_book(
        book,
        tmp_path,
        fetch_many=_fake_fetch({"https://courtlistener.com/": (200, None)}),
        proxy="socks5://egress:1080",
    )
    assert load_reachability(tmp_path) is not None
    assert first.proxy == "socks5://egress:1080"
    assert first.hosts["courtlistener.com"].state == "reachable"
    assert first.hosts["nycourts.gov"].state == "unknown"  # no response in the fake

    # A later probe with an extra host keeps earlier verdicts for hosts it re-probes AND merges.
    second = probe_book(
        book,
        tmp_path,
        extra_hosts=["https://gao.gov/reports"],
        fetch_many=_fake_fetch(
            {"https://gao.gov/": (403, None), "https://courtlistener.com/": (200, None)}
        ),
    )
    assert second.hosts["gao.gov"].state == "blocked"
    assert second.hosts["courtlistener.com"].state == "reachable"
    text = render_reachability(second)
    assert "blocked" in text and "gao.gov" in text and "1 blocked" in text


def test_load_reachability_tolerates_garbage(tmp_path, capsys):
    (tmp_path / "reachability.json").write_text("{not json")
    assert load_reachability(tmp_path) is None
    assert "re-run" in capsys.readouterr().out


def test_probe_subcommand_no_hosts(tmp_path, capsys):
    cfg = tmp_path / "book.yaml"
    cfg.write_text(
        "title: T\ndescription: D\nchapters:\n  - number: 1\n    title: A\n"
        "    description: B\n    research_questions: ['What happened, with dates?']\n"
    )
    assert main_mod.main(["probe", str(cfg)]) == 0
    assert "No hosts to probe" in capsys.readouterr().out
