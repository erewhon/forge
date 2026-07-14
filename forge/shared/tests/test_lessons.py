"""Rules-layer lessons file — read/render, budgeted append with eviction, and propose-not-append."""

from __future__ import annotations

from pathlib import Path

from forge.shared.lessons import (
    append_lesson,
    draft_lesson,
    lesson_bullets,
    lessons_path,
    propose_lesson,
    proposed_lessons,
    read_lessons,
    render_lessons_preamble,
)


def test_read_lessons_absent_is_empty(tmp_path: Path):
    assert read_lessons(tmp_path) == ""


def test_bullets_ignores_headers_and_prose():
    text = "# Repository lessons\n\nsome prose\n- first lesson\n- second lesson\n"
    assert lesson_bullets(text) == ["- first lesson", "- second lesson"]


def test_render_preamble_empty_when_no_lessons():
    assert render_lessons_preamble("") == ""
    assert render_lessons_preamble("# header only\n\njust prose\n") == ""


def test_render_preamble_wraps_the_bullets():
    out = render_lessons_preamble("- always run the formatter\n- never touch generated files\n")
    assert "READ FIRST" in out
    assert "- always run the formatter" in out
    assert "- never touch generated files" in out
    assert out.rstrip().endswith("---")  # separated from the task that follows


def test_append_lesson_writes_and_dedupes(tmp_path: Path):
    assert append_lesson(tmp_path, "pin the toolchain version") is True
    assert lessons_path(tmp_path).is_file()
    assert append_lesson(tmp_path, "pin the toolchain version") is False  # already present
    assert append_lesson(tmp_path, "- pin the toolchain version") is False  # bullet form == same
    assert lesson_bullets(read_lessons(tmp_path)) == ["- pin the toolchain version"]


def test_append_lesson_evicts_oldest_over_budget(tmp_path: Path):
    for i in range(5):
        append_lesson(tmp_path, f"lesson {i}", max_lessons=3)
    bullets = lesson_bullets(read_lessons(tmp_path))
    assert bullets == ["- lesson 2", "- lesson 3", "- lesson 4"]  # oldest two fell off


def test_draft_lesson_is_one_line_and_cites_the_count():
    lesson = draft_lesson("gate: pytest failed\nsome traceback\nmore", count=3)
    assert lesson.count("\n") == 0
    assert "3×" in lesson
    assert "gate: pytest failed" in lesson


def test_propose_lesson_writes_visible_artifact_and_dedupes(tmp_path: Path):
    assert propose_lesson(tmp_path, "consider adding a smoke test") is True
    assert propose_lesson(tmp_path, "consider adding a smoke test") is False
    assert proposed_lessons(tmp_path) == ["- consider adding a smoke test"]


def test_propose_never_touches_the_active_lessons_file(tmp_path: Path):
    propose_lesson(tmp_path, "some candidate")
    assert read_lessons(tmp_path) == ""  # active file untouched — a human promotes proposals
