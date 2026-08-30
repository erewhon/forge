"""The brief: guidance and source policy reach the planner and researcher prompts (no network)."""

from __future__ import annotations

from forge.book_researcher import planner, researcher
from forge.book_researcher.brief import (
    effective_policy,
    render_book_brief,
    render_chapter_brief,
    render_chapter_line,
)
from forge.book_researcher.models import (
    BookConfig,
    ChapterOutline,
    HostVerdict,
    Reachability,
    SourcePolicy,
    SprintContract,
)


def _book() -> BookConfig:
    return BookConfig(
        title="T",
        description="A thesis.",
        guidance=["Distinguish adjudicated from alleged."],
        sources=SourcePolicy(reachable=["courtlistener.com"], blocked=["nycourts.gov"]),
        chapters=[
            ChapterOutline(
                number=1,
                title="One",
                description="What chapter one establishes.",
                sources=["efts.sec.gov"],
                guidance=["Never supply a docket number a source did not show."],
                research_questions=["Q1?", "Q2?"],
            )
        ],
    )


def _reach(**hosts: str) -> Reachability:
    return Reachability(
        probed_at="now",
        hosts={
            h: HostVerdict(host=h, url=f"https://{h}/", status=None, state=s)  # type: ignore[arg-type]
            for h, s in hosts.items()
        },
    )


def test_effective_policy_merges_probe_but_human_blocked_wins():
    book = _book()
    reach = _reach(**{"gao.gov": "blocked", "fec.gov": "reachable", "nycourts.gov": "reachable"})
    policy = effective_policy(book, reach)
    assert "gao.gov" in policy.blocked
    assert "fec.gov" in policy.reachable
    # human said blocked; a 200 at the root does not override that
    assert "nycourts.gov" in policy.blocked and "nycourts.gov" not in policy.reachable
    assert policy.reachable[0] == "courtlistener.com"  # yaml entries keep their order first


def test_effective_policy_without_probe_is_the_yaml_policy():
    book = _book()
    assert effective_policy(book, None) == book.sources


def test_chapter_brief_carries_everything_the_question_used_to():
    book = _book()
    text = render_chapter_brief(book, book.chapters[0], book.sources)
    assert "Chapter 1: One" in text and "What chapter one establishes." in text
    assert "Distinguish adjudicated" in text and "docket number" in text
    assert "efts.sec.gov" in text
    assert "nycourts.gov" in text and "NOT reachable" in text
    assert "courtlistener.com" in text


def test_book_brief_and_chapter_line():
    book = _book()
    assert "Book-wide research rules" in render_book_brief(book, book.sources)
    line = render_chapter_line(book.chapters[0])
    assert "Primary sources: efts.sec.gov" in line and "Chapter rules:" in line


def test_researcher_prompt_includes_brief(monkeypatch):
    seen: dict[str, str] = {}

    def fake_complete(cfg, *, system, user_message, model, max_tokens=4096, retries=1):
        seen["user"] = user_message
        return '{"question": "Q1?", "answer": "A", "sources": [], "confidence": "low"}'

    monkeypatch.setattr(researcher, "complete_with_retry", fake_complete)
    monkeypatch.setattr(researcher.settings, "check_sources", False)
    contract = SprintContract(
        sprint_id="001", chapter=1, questions=["Q1?"], success_criteria=[], priority="high"
    )
    researcher.execute_sprint(contract, brief="THE BRIEF")
    assert seen["user"].startswith("THE BRIEF")
    assert "Research question: Q1?" in seen["user"]


def test_planner_prompt_includes_book_brief_and_policy(monkeypatch, tmp_path):
    seen: dict[str, str] = {}

    def fake_complete(cfg, *, system, user_message, model, max_tokens=4096):
        seen["user"] = user_message
        seen["system"] = system
        return '{"chapter": 1, "questions": ["Q1?"], "success_criteria": ["c"], "priority": "high"}'

    monkeypatch.setattr(planner, "complete", fake_complete)
    monkeypatch.setattr(planner.settings, "project_dir", tmp_path)
    book = _book()
    policy = SourcePolicy(blocked=["gao.gov"])
    planner.create_sprint(book, {}, 1, policy=policy)
    assert "Distinguish adjudicated" in seen["user"]
    assert "gao.gov" in seen["user"]
    assert "Primary sources: efts.sec.gov" in seen["user"]
    assert "unreachable" in seen["system"]
