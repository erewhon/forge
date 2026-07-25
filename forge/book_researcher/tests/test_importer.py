"""Tests for `forge book import-research`: transplant a research topic into a chapter's knowledge.

Both project dirs are redirected to tmp paths — the research dir via the env var (highest
precedence, so it beats the repo .env) and the book dir by patching the settings singleton.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.book_researcher import importer
from forge.book_researcher.models import SprintFindings
from forge.general_researcher.models import ResearchFinding as GRFinding
from forge.general_researcher.models import SprintFindings as GRSprint


def _write_research_topic(root: Path, slug: str, sprints: list[GRSprint]) -> None:
    findings_dir = root / slug / "findings"
    findings_dir.mkdir(parents=True)
    for sf in sprints:
        (findings_dir / f"sprint-{sf.sprint_id}.json").write_text(sf.model_dump_json(indent=2))


@pytest.fixture
def dirs(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    research_root = tmp_path / "research"
    book_root = tmp_path / "book"
    research_root.mkdir()
    book_root.mkdir()
    # research dir: env var wins over the repo .env; book dir: patch the singleton
    monkeypatch.setenv("GENERAL_RESEARCHER_PROJECT_DIR", str(research_root))
    monkeypatch.setattr(importer.settings, "project_dir", book_root)
    return research_root, book_root


def test_import_transplants_findings_and_stamps_chapter(dirs):
    research_root, book_root = dirs
    _write_research_topic(
        research_root,
        "why-sky-blue",
        [
            GRSprint(
                sprint_id="001",
                findings=[
                    GRFinding(
                        question="What is Rayleigh scattering?",
                        answer="A scatters B.",
                        sources=["wiki"],
                        confidence="high",
                    ),
                    GRFinding(
                        question="Bad Q",
                        answer="Research failed: empty response",
                        sources=[],
                        confidence="low",
                    ),
                ],
            ),
        ],
    )

    result = importer.import_research("why-sky-blue", chapter=3)

    assert result.imported == 1  # the failed finding is dropped
    assert result.dropped == 1
    assert (
        result.json_path
        == book_root / "knowledge" / "chapter-03" / "sprint-import-why-sky-blue.json"
    )
    assert result.md_path.exists()

    seed = SprintFindings.model_validate(json.loads(result.json_path.read_text()))
    assert seed.chapter == 3
    assert seed.sprint_id == "import-why-sky-blue"
    assert [f.question for f in seed.findings] == ["What is Rayleigh scattering?"]
    assert seed.findings[0].sources == ["wiki"]


def test_import_dedupes_questions_later_sprint_wins(dirs):
    research_root, _ = dirs
    _write_research_topic(
        research_root,
        "topic",
        [
            GRSprint(
                sprint_id="001",
                findings=[
                    GRFinding(question="Q", answer="first pass", sources=[], confidence="low")
                ],
            ),
            GRSprint(
                sprint_id="002",
                findings=[
                    GRFinding(question="Q", answer="refined", sources=["s"], confidence="high")
                ],
            ),
        ],
    )

    result = importer.import_research("topic", chapter=1)

    assert result.imported == 1  # deduped to a single question
    seed = SprintFindings.model_validate(json.loads(result.json_path.read_text()))
    assert seed.findings[0].answer == "refined"  # later sprint wins
    assert seed.findings[0].confidence == "high"


def test_unknown_slug_raises_with_available_hint(dirs):
    research_root, _ = dirs
    _write_research_topic(
        research_root,
        "real-topic",
        [
            GRSprint(
                sprint_id="001",
                findings=[GRFinding(question="Q", answer="a", sources=[], confidence="low")],
            )
        ],
    )
    with pytest.raises(FileNotFoundError, match="real-topic"):
        importer.import_research("missing-topic", chapter=1)


def test_all_failed_findings_raises(dirs):
    research_root, _ = dirs
    _write_research_topic(
        research_root,
        "dead-topic",
        [
            GRSprint(
                sprint_id="001",
                findings=[
                    GRFinding(
                        question="Q",
                        answer="Research failed: empty response",
                        sources=[],
                        confidence="low",
                    )
                ],
            ),
        ],
    )
    with pytest.raises(ValueError, match="no usable findings"):
        importer.import_research("dead-topic", chapter=1)
