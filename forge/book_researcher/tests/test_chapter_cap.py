"""One unpassable chapter must not absorb the whole run.

Observed live: a ten-sprint run put SIX consecutive sprints into chapter 2 while chapters 3-10
received no research at all. A failing sprint's feedback is handed to the next planner call, which
pushes it to re-attack the same chapter, so a chapter that *cannot* pass starves the rest of the
outline. After MAX_ATTEMPTS_PER_CHAPTER failures the chapter is declared exhausted and the planner
is explicitly told to go elsewhere.
"""

from __future__ import annotations

from forge.book_researcher.models import BookConfig, ChapterOutline
from forge.book_researcher.planner import create_sprint


def _book() -> BookConfig:
    return BookConfig(
        title="T",
        description="D",
        chapters=[
            ChapterOutline(number=n, title=f"C{n}", description="d", research_questions=["q"])
            for n in (1, 2, 3)
        ],
    )


def _capture_user_msg(monkeypatch) -> dict:
    seen: dict = {}

    def fake_complete(cfg, *, system, user_message, model, max_tokens=4096):
        seen["user"] = user_message
        return '{"chapter": 3, "questions": ["q"], "success_criteria": [], "priority": "high"}'

    monkeypatch.setattr("forge.book_researcher.planner.complete", fake_complete)
    return seen


def test_exhausted_chapter_is_marked_and_forbidden(monkeypatch):
    seen = _capture_user_msg(monkeypatch)
    create_sprint(_book(), {}, 4, follow_up_feedback="failed again", exhausted_chapters={2})
    msg = seen["user"]
    assert "ATTEMPT LIMIT REACHED" in msg
    assert "CHAPTERS 2 HAVE REACHED THEIR ATTEMPT LIMIT" in msg
    assert "Choose a different chapter" in msg
    # the follow-up instruction must also redirect rather than say "address these gaps" flatly
    assert "IN A DIFFERENT CHAPTER" in msg


def test_no_exhaustion_leaves_prompt_clean(monkeypatch):
    seen = _capture_user_msg(monkeypatch)
    create_sprint(_book(), {}, 2, follow_up_feedback="failed once")
    msg = seen["user"]
    assert "ATTEMPT LIMIT" not in msg
    assert "IN A DIFFERENT CHAPTER" not in msg
    assert "Create a follow-up sprint addressing these gaps." in msg


def test_multiple_exhausted_chapters_all_listed(monkeypatch):
    seen = _capture_user_msg(monkeypatch)
    create_sprint(_book(), {}, 6, exhausted_chapters={1, 2})
    assert "CHAPTERS 1, 2 HAVE REACHED" in seen["user"]
