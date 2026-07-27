"""Prior-sprint context must not launder fabrications into fact.

Regression cover for the worst failure this harness has produced. A court docket number entered
chapter 2 in sprint 1 with ZERO sources. The whole rendered markdown was fed to the next sprint as
"existing research context", so sprint 2 restated the docket at HIGH confidence — attaching sources
it had found for adjacent facts — and it survived four further sprints unchallenged. An invention
acquired credibility purely by being repeated inside the harness's own memory.

Two defences, both pinned here: unsourced/failed findings never propagate, and whatever does is
labelled an unverified claim rather than an established fact.
"""

from __future__ import annotations

import json
from pathlib import Path

from forge.book_researcher import main as book_main
from forge.book_researcher.models import ResearchFinding, SprintFindings
from forge.shared.llm import RESEARCH_FAILED_PREFIX

FABRICATED = "The docket number is APL-24-2299 and leave was denied on June 18, 2026."
SOURCED = "The court found persistent fraud in asset valuations."


def _write(
    tmp: Path, chapter: int, findings: list[ResearchFinding], sprint_id: str = "001"
) -> None:
    d = tmp / "knowledge" / f"chapter-{chapter:02d}"
    d.mkdir(parents=True, exist_ok=True)
    sf = SprintFindings(sprint_id=sprint_id, chapter=chapter, findings=findings)
    (d / f"sprint-{sprint_id}.json").write_text(sf.model_dump_json(indent=2))


def test_unsourced_finding_does_not_propagate(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(book_main.settings, "project_dir", tmp_path)
    _write(
        tmp_path,
        2,
        [
            # the sprint-1 fabrication: confident prose, no sources
            ResearchFinding(question="docket?", answer=FABRICATED, sources=[], confidence="low"),
            ResearchFinding(
                question="holding?", answer=SOURCED, sources=["CourtListener"], confidence="high"
            ),
        ],
    )
    ctx = book_main._get_chapter_context(2)
    assert "APL-24-2299" not in ctx, "an unsourced claim must never be fed forward"
    assert SOURCED in ctx


def test_failed_finding_does_not_propagate(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(book_main.settings, "project_dir", tmp_path)
    _write(
        tmp_path,
        3,
        [
            ResearchFinding(
                question="q",
                answer=f"{RESEARCH_FAILED_PREFIX} 11 of 12 tool call(s) failed",
                sources=[],
                confidence="low",
            )
        ],
    )
    assert book_main._get_chapter_context(3) == ""


def test_surviving_context_is_labelled_unverified(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(book_main.settings, "project_dir", tmp_path)
    _write(
        tmp_path,
        4,
        [ResearchFinding(question="q", answer=SOURCED, sources=["AP"], confidence="high")],
    )
    ctx = book_main._get_chapter_context(4)
    assert "UNVERIFIED CLAIMS" in ctx
    assert "not established facts" in ctx
    # and it must tell the model to re-verify rather than cite the context itself
    assert "verify it against a source you retrieve yourself" in ctx


def test_missing_chapter_is_empty(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(book_main.settings, "project_dir", tmp_path)
    assert book_main._get_chapter_context(9) == ""


def test_malformed_sprint_file_is_skipped(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(book_main.settings, "project_dir", tmp_path)
    d = tmp_path / "knowledge" / "chapter-05"
    d.mkdir(parents=True)
    (d / "sprint-001.json").write_text("{not json")
    _write(
        tmp_path,
        5,
        [ResearchFinding(question="q", answer=SOURCED, sources=["AP"], confidence="high")],
        sprint_id="002",
    )
    ctx = book_main._get_chapter_context(5)
    assert SOURCED in ctx  # the good file still loads


def test_json_is_the_source_of_truth_not_markdown(monkeypatch, tmp_path: Path):
    """A stale/rendered .md must not reintroduce what the JSON filter removed."""
    monkeypatch.setattr(book_main.settings, "project_dir", tmp_path)
    _write(
        tmp_path,
        6,
        [ResearchFinding(question="q", answer=FABRICATED, sources=[], confidence="low")],
    )
    (tmp_path / "knowledge" / "chapter-06" / "sprint-001.md").write_text(
        f"# Sprint 001\n\n{FABRICATED}\n"
    )
    assert "APL-24-2299" not in book_main._get_chapter_context(6)
    assert json  # keep the import meaningful for readers of this file
