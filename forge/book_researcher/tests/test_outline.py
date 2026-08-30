"""Outline helpers: fuzzy question matching and the YAML round-trip."""

from __future__ import annotations

import yaml

from forge.book_researcher.models import BookConfig, ChapterOutline, SourcePolicy
from forge.book_researcher.outline import match_question, render_outline_yaml
from forge.book_researcher.renderer import render_knowledge_summary


def test_match_question_exact_fuzzy_substring_and_miss():
    qs = [
        "What findings of fact did Justice Engoron make about asset valuation, per the decision?",
        "What is the strongest argument that the valuations were ordinary practice?",
    ]
    assert match_question(qs[0], qs) == (0, 1.0)
    assert (
        match_question("what findings of fact did justice engoron make about asset valuation", qs)[
            0
        ]
        == 0
    )
    assert (
        match_question(
            "What findings of fact did Justice Engoron make about asset valuations, per the "
            "ruling?",
            qs,
        )[0]
        == 0
    )
    assert match_question("Who pardoned CZ and when?", qs) is None
    assert match_question("", qs) is None


def test_render_outline_yaml_round_trips_and_omits_empty_optionals():
    book = BookConfig(
        title="T",
        description="Line one.\nLine two.",
        sources=SourcePolicy(blocked=["gao.gov"]),
        chapters=[
            ChapterOutline(
                number=1,
                title="One",
                description="D",
                guidance=["rule"],
                research_questions=["Q1?", "Q2?"],
            )
        ],
    )
    text = render_outline_yaml(book, header="Provenance line\nsecond line")
    assert text.startswith("# Provenance line\n# second line\n\n")
    assert "reachable" not in text and "notes" not in text  # empty policy lists omitted
    assert "sources:" in text and "guidance:" in text
    assert BookConfig.model_validate(yaml.safe_load(text)) == book


def test_knowledge_summary_gaps_are_fuzzy(tmp_path):
    from forge.book_researcher.models import ResearchFinding, SprintFindings

    book = BookConfig(
        title="T",
        description="D",
        chapters=[
            ChapterOutline(
                number=1,
                title="One",
                description="D",
                research_questions=[
                    "What findings of fact did Justice Engoron make about asset valuation?",
                    "Who pardoned CZ, and when, per the Federal Register?",
                ],
            )
        ],
    )
    ch = tmp_path / "chapter-01"
    ch.mkdir()
    sf = SprintFindings(
        sprint_id="001",
        chapter=1,
        findings=[
            ResearchFinding(
                question="What findings of fact did Justice Engoron make about asset valuations?",
                answer="a",
                sources=["s"],
                confidence="high",
            )
        ],
    )
    (ch / "sprint-001.json").write_text(sf.model_dump_json())
    summary = render_knowledge_summary(book, tmp_path)
    assert "Who pardoned CZ" in summary  # still a gap
    assert "Justice Engoron" not in summary.split("Remaining gaps")[1]  # rephrased, but covered
