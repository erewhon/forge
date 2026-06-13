from __future__ import annotations

import json
from pathlib import Path

from agents.book_researcher.models import (
    BookConfig,
    SprintFindings,
    VerificationResult,
)


def render_sprint_findings(findings: SprintFindings) -> str:
    """Render sprint findings as readable markdown."""
    lines = [
        f"# Sprint {findings.sprint_id} - Chapter {findings.chapter}",
        "",
    ]

    for f in findings.findings:
        lines.extend([
            f"## {f.question}",
            "",
            f"{f.answer}",
            "",
            f"**Confidence:** {f.confidence}",
            "",
        ])
        if f.sources:
            lines.append("**Sources:**")
            for src in f.sources:
                lines.append(f"- {src}")
            lines.append("")
        else:
            lines.append("**Sources:** None cited")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def render_verification(result: VerificationResult) -> str:
    """Render verification scores as a formatted table with feedback."""
    s = result.scores
    status = "PASSED" if result.passed else "FAILED"

    lines = [
        f"# Verification: Sprint {result.sprint_id} [{status}]",
        "",
        "| Criterion | Score |",
        "|-----------|-------|",
        f"| Source Diversity | {s.source_diversity}/10 |",
        f"| Claim Verification | {s.claim_verification}/10 |",
        f"| Counter-Narrative | {s.counter_narrative}/10 |",
        f"| Depth | {s.depth}/10 |",
        f"| Actionability | {s.actionability}/10 |",
        f"| **Overall** | **{s.overall}/10** |",
        "",
        "## Feedback",
        "",
        result.feedback,
        "",
    ]

    if result.follow_up_questions:
        lines.extend([
            "## Follow-up Questions",
            "",
        ])
        for q in result.follow_up_questions:
            lines.append(f"- {q}")
        lines.append("")

    return "\n".join(lines)


def render_knowledge_summary(book_config: BookConfig, knowledge_dir: Path) -> str:
    """Produce a summary of all research across chapters."""
    lines = [
        f"# Research Summary: {book_config.title}",
        "",
        f"{book_config.description}",
        "",
        "---",
        "",
    ]

    total_findings = 0
    chapters_with_research = 0
    all_scores: list[int] = []

    for ch in book_config.chapters:
        chapter_dir = knowledge_dir / f"chapter-{ch.number:02d}"
        sprint_files = sorted(chapter_dir.glob("sprint-*.json")) if chapter_dir.exists() else []
        # Filter out non-findings JSON (e.g. review files in sprints_dir would be elsewhere)
        sprint_files = [f for f in sprint_files if f.name.startswith("sprint-")]

        if not sprint_files:
            lines.extend([
                f"## Chapter {ch.number}: {ch.title}",
                "",
                "*No research yet.*",
                "",
            ])
            continue

        chapters_with_research += 1
        chapter_finding_count = 0
        questions_covered: list[str] = []

        for sf in sprint_files:
            try:
                data = json.loads(sf.read_text())
                sf_parsed = SprintFindings.model_validate(data)
                chapter_finding_count += len(sf_parsed.findings)
                questions_covered.extend(f.question for f in sf_parsed.findings)
            except Exception:
                continue

        total_findings += chapter_finding_count

        # Check for reviews in sprints dir
        sprints_dir = knowledge_dir.parent / "sprints"
        review_files = sorted(sprints_dir.glob("sprint-*-review.json")) if sprints_dir.exists() else []
        chapter_scores: list[int] = []
        for rf in review_files:
            try:
                review_data = json.loads(rf.read_text())
                result = VerificationResult.model_validate(review_data)
                # Match reviews to this chapter's sprints
                for sf in sprint_files:
                    if result.sprint_id in sf.name:
                        chapter_scores.append(result.scores.overall)
                        all_scores.append(result.scores.overall)
                        break
            except Exception:
                continue

        avg_score = sum(chapter_scores) / len(chapter_scores) if chapter_scores else 0

        lines.extend([
            f"## Chapter {ch.number}: {ch.title}",
            "",
            f"- **Sprints completed:** {len(sprint_files)}",
            f"- **Findings:** {chapter_finding_count}",
            f"- **Average score:** {avg_score:.1f}/10" if chapter_scores else "- **Average score:** N/A",
            f"- **Questions covered:** {len(questions_covered)}",
            "",
        ])

        # Show coverage gaps
        covered_set = {q.lower().strip() for q in questions_covered}
        gaps = [q for q in ch.research_questions if q.lower().strip() not in covered_set]
        if gaps:
            lines.append("**Remaining gaps:**")
            for g in gaps:
                lines.append(f"- {g}")
            lines.append("")

    # Summary stats
    avg_overall = sum(all_scores) / len(all_scores) if all_scores else 0
    total_chapters = len(book_config.chapters)

    lines.extend([
        "---",
        "",
        "## Overall Statistics",
        "",
        f"- **Chapters with research:** {chapters_with_research}/{total_chapters}",
        f"- **Total findings:** {total_findings}",
        f"- **Average verification score:** {avg_overall:.1f}/10" if all_scores else "- **Average verification score:** N/A",
        f"- **Chapters needing research:** {total_chapters - chapters_with_research}",
        "",
    ])

    return "\n".join(lines)
