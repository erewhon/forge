"""Pull a `forge research` topic's findings into a book chapter's knowledge dir.

`forge research` writes scoped-topic findings under `GENERAL_RESEARCHER_PROJECT_DIR/<slug>/`; this
transplants the usable ones into a book's `knowledge/chapter-NN/` so `forge book` treats them as
prior context — scanned for coverage (it won't re-ask those questions) and fed to the researcher on
the next sprint — instead of re-researching from scratch.

The two harnesses share the `ResearchFinding` shape (question/answer/sources/confidence), so the
transform only adds the book's `chapter` field and drops empty/failed findings. The seed is written
with a non-numeric sprint id (`import-<slug>`) so it counts as covered knowledge without inflating
the book's sprint counter (which only globs `sprint-[0-9]*` in the sprints dir).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from forge.book_researcher.config import settings
from forge.book_researcher.models import ResearchFinding, SprintFindings
from forge.book_researcher.renderer import render_sprint_findings
from forge.general_researcher.config import GeneralResearcherSettings
from forge.general_researcher.models import SprintFindings as ResearchSprintFindings
from forge.shared.llm import RESEARCH_FAILED_PREFIX


@dataclass(frozen=True)
class ImportResult:
    json_path: Path
    md_path: Path
    imported: int
    dropped: int
    source_dir: Path


def _usable(answer: str) -> bool:
    """A finding worth importing: non-empty answer that isn't a failure marker."""
    return bool(answer.strip()) and not answer.startswith(RESEARCH_FAILED_PREFIX)


def _available_slugs() -> list[str]:
    root = GeneralResearcherSettings().project_dir
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "findings").is_dir())


def import_research(slug: str, chapter: int) -> ImportResult:
    """Transplant research topic `slug`'s findings into book `chapter`'s knowledge dir.

    Raises FileNotFoundError if the topic has no findings dir, or ValueError if every finding is
    empty/failed (nothing worth importing). Returns where it wrote and how much it pulled in.
    """
    research_dir = GeneralResearcherSettings().project_dir / slug
    findings_dir = research_dir / "findings"
    if not findings_dir.is_dir():
        available = _available_slugs()
        hint = f" Available topics: {', '.join(available)}." if available else ""
        raise FileNotFoundError(
            f"no research findings at {findings_dir} (topic slug {slug!r} not found).{hint}"
        )

    # Collect usable findings across all the topic's sprints; dedupe by question with later sprints
    # winning (later sprints refine earlier answers). dict preserves first-seen order per key.
    by_question: dict[str, ResearchFinding] = {}
    dropped = 0
    for jf in sorted(findings_dir.glob("sprint-*.json")):
        try:
            sf = ResearchSprintFindings.model_validate(json.loads(jf.read_text()))
        except Exception:
            continue
        for f in sf.findings:
            if not _usable(f.answer):
                dropped += 1
                continue
            by_question[f.question.strip().lower()] = ResearchFinding(
                question=f.question,
                answer=f.answer,
                sources=f.sources,
                confidence=f.confidence,
            )

    if not by_question:
        raise ValueError(
            f"research topic {slug!r} has no usable findings to import "
            "(all empty/failed) — nothing written."
        )

    seed = SprintFindings(
        sprint_id=f"import-{slug}",
        chapter=chapter,
        findings=list(by_question.values()),
        raw_search_notes=f"Imported from general_researcher topic '{slug}' ({research_dir}).",
    )

    chapter_dir = settings.knowledge_dir / f"chapter-{chapter:02d}"
    chapter_dir.mkdir(parents=True, exist_ok=True)
    json_path = chapter_dir / f"sprint-import-{slug}.json"
    md_path = chapter_dir / f"sprint-import-{slug}.md"
    json_path.write_text(seed.model_dump_json(indent=2))
    md_path.write_text(render_sprint_findings(seed))

    return ImportResult(
        json_path=json_path,
        md_path=md_path,
        imported=len(by_question),
        dropped=dropped,
        source_dir=research_dir,
    )
