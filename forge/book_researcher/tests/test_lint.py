"""`forge book lint`: the deterministic rules, the report, and the mocked critic."""

from __future__ import annotations

from types import SimpleNamespace

import yaml

from forge.book_researcher import lint as lint_mod
from forge.book_researcher import main as main_mod
from forge.book_researcher.lint import (
    critique_chapter,
    critiques_to_findings,
    has_errors,
    lint_book,
    render_lint_report,
)
from forge.book_researcher.models import (
    BookConfig,
    ChapterCritique,
    ChapterOutline,
    QuestionCritique,
    SourcePolicy,
)
from forge.book_researcher.scaffold import BOOK_SKELETON


def _chapter(number: int, questions: list[str], **kw) -> ChapterOutline:
    return ChapterOutline(
        number=number,
        title=f"Chapter {number} title",
        description="What this chapter must establish, in a sentence or two.",
        research_questions=questions,
        **kw,
    )


def _book(*chapters: ChapterOutline, **kw) -> BookConfig:
    return BookConfig(
        title="A Real Title",
        description="A thesis sentence long enough to not be thin at all.",
        chapters=list(chapters),
        **kw,
    )


GOOD = [
    "What findings of fact did Justice Engoron make about asset valuation, per the decision text?",
    "What is the strongest legal argument that the valuations were ordinary practice, and who "
    "advances it?",
]


def _rules(findings) -> set[str]:
    return {f.rule for f in findings}


def test_good_chapter_is_clean():
    assert lint_book(_book(_chapter(1, GOOD))) == []


def test_skeleton_has_placeholder_errors():
    book = BookConfig.model_validate(yaml.safe_load(BOOK_SKELETON))
    findings = lint_book(book)
    assert has_errors(findings)
    assert "placeholder" in _rules(findings)


def test_no_counter_narrative_and_no_source_demand():
    qs = [
        "What did Justice Engoron decide about Mar-a-Lago's valuation in 2023?",
        "How much did Trump Organization pay in disgorgement in 2024?",
    ]
    rules = _rules(lint_book(_book(_chapter(1, qs))))
    assert "no-counter-narrative" in rules
    assert "no-source-demand" in rules
    # chapter sources satisfy the source demand
    rules = _rules(lint_book(_book(_chapter(1, qs, sources=["courtlistener.com"]))))
    assert "no-source-demand" not in rules


def test_counter_narrative_vocabulary_is_broad():
    for q in [
        "What is the argument that these removals were lawful, per the OLC memo?",
        "What is the base rate for such trading absent inside information, per CFTC data?",
        "How do Lutnick's disclosures compare with those of prior Commerce secretaries, per OGE?",
    ]:
        assert "no-counter-narrative" not in _rules(lint_book(_book(_chapter(1, [GOOD[0], q]))))


def test_compound_only_on_genuinely_multiple_questions():
    two_part = "In People v. Trump, what did the court hold, and on what evidence, per the opinion?"
    assert "compound-question" not in _rules(lint_book(_book(_chapter(1, [two_part, GOOD[1]]))))
    three = (
        "What did the Appellate Division hold on appeal — which parts were vacated, which "
        "upheld, and on what reasoning, per the opinion?"
    )
    assert "compound-question" in _rules(lint_book(_book(_chapter(1, [three, GOOD[1]]))))
    two_marks = "What did Engoron hold per the opinion? And who argues it was wrong?"
    assert "compound-question" in _rules(lint_book(_book(_chapter(1, [two_marks, GOOD[1]]))))


def test_no_explicit_subject_flags_demonstratives_not_definite_nouns():
    demonstrative = "What is the argument that these removals were lawful, per the memo?"
    f = [x for x in lint_book(_book(_chapter(1, [GOOD[0], demonstrative])))]
    hit = [x for x in f if x.rule == "no-explicit-subject"]
    assert hit and "these removals" in hit[0].message
    # A proper noun or a digit resolves it
    named = "What is the argument that the 2025 OGE removals were lawful, per the memo?"
    assert "no-explicit-subject" not in _rules(lint_book(_book(_chapter(1, [GOOD[0], named]))))
    # "the case" is resolved by the chapter brief now — not flagged
    definite = "What did the case establish about valuation, per the decision text?"
    assert "no-explicit-subject" not in _rules(lint_book(_book(_chapter(1, [GOOD[1], definite]))))


def test_vague_question():
    for q in ["Tell me about emoluments.", "What is corruption?", "Discuss pardons etc."]:
        assert "vague-question" in _rules(lint_book(_book(_chapter(1, [q, GOOD[1]]))))


def test_blocked_host_named_unless_negated():
    policy = SourcePolicy(blocked=["nycourts.gov"])
    uses = "What did the First Department hold, per the opinion on nycourts.gov?"
    warns = "What did the First Department hold? Use CourtListener; nycourts.gov is unreachable."
    assert "blocked-host-named" in _rules(lint_book(_book(_chapter(1, [uses, GOOD[1]])), policy))
    assert "blocked-host-named" not in _rules(
        lint_book(_book(_chapter(1, [warns, GOOD[1]])), policy)
    )


def test_instruction_in_question():
    long_q = (
        "Has the New York Court of Appeals acted on People v. Trump? Report ONLY what a retrieved "
        "source states, quoting it. Do NOT supply a docket number, case index, or filing date "
        "unless it appears verbatim in a source you retrieved — an invented docket is worse than "
        "reporting that the record could not be found."
    )
    assert "instruction-in-question" in _rules(lint_book(_book(_chapter(1, [long_q, GOOD[1]]))))


def test_structural_errors():
    dup = _book(_chapter(1, GOOD), _chapter(1, GOOD))
    rules = _rules(lint_book(dup))
    assert "chapter-numbering" in rules and "duplicate-question" in rules
    assert has_errors(lint_book(dup))
    empty = _book(_chapter(1, []))
    assert "no-questions" in _rules(lint_book(empty))
    many = _book(
        _chapter(
            1, GOOD + [f"What did Engoron hold about item {i}, per the opinion?" for i in range(5)]
        )
    )
    assert "too-many-questions" in _rules(lint_book(many))


def test_report_groups_by_chapter_and_marks_errors():
    book = _book(_chapter(1, []), _chapter(2, GOOD))
    text = render_lint_report(book, lint_book(book))
    assert "1 errors" in text and "chapter 1: Chapter 1 title" in text
    assert "ERROR [no-questions]" in text
    assert render_lint_report(book, []).endswith("clean")


def test_critique_chapter_uses_structured_and_filters_by_threshold(monkeypatch):
    book = _book(_chapter(1, GOOD))
    value = ChapterCritique(
        chapter=1,
        critiques=[
            QuestionCritique(
                question=GOOD[0],
                weakest_dimension="depth",
                predicted_score=5,
                problem="asks for findings without a period",
                rewrite="What findings of fact about 2011-2015 valuations did Engoron make?",
            ),
            QuestionCritique(
                question=GOOD[1], weakest_dimension="depth", predicted_score=8, problem="fine"
            ),
        ],
    )
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        assert kwargs["predicate"](value)  # echoing the questions satisfies the predicate
        return SimpleNamespace(value=value, error=None, ok=True, raw="")

    monkeypatch.setattr(lint_mod, "structured", fake)
    result = critique_chapter(book, book.chapters[0], book.sources, pool=object())  # type: ignore[arg-type]
    assert result is value
    assert "Chapter 1" in calls[0]["user"] and GOOD[0] in calls[0]["user"]
    findings = critiques_to_findings([value], threshold=7)
    assert len(findings) == 1 and findings[0].rule == "critic" and "rewrite:" in findings[0].message


def test_lint_subcommand_exit_codes(tmp_path):
    good = tmp_path / "good.yaml"
    good.write_text(yaml.safe_dump(_book(_chapter(1, GOOD)).model_dump()))
    assert main_mod.main(["lint", str(good)]) == 0
    bad = tmp_path / "bad.yaml"
    bad.write_text(BOOK_SKELETON)
    assert main_mod.main(["lint", str(bad)]) == 1
    warn_only = tmp_path / "warn.yaml"
    warn_only.write_text(
        yaml.safe_dump(
            _book(_chapter(1, ["Tell me about Engoron's ruling.", GOOD[1]])).model_dump()
        )
    )
    assert main_mod.main(["lint", str(warn_only)]) == 0
    assert main_mod.main(["lint", str(warn_only), "--strict"]) == 1
    assert main_mod.main(["lint", str(tmp_path / "missing.yaml")]) == 1
