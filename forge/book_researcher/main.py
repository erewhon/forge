from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from agents.book_researcher.config import settings
from agents.book_researcher.models import BookConfig, SprintFindings
from agents.book_researcher.planner import create_sprint
from agents.book_researcher.renderer import render_knowledge_summary, render_verification
from agents.book_researcher.researcher import execute_sprint
from agents.book_researcher.verifier import verify_sprint


def _load_book_config(config_path: str) -> BookConfig:
    """Load book configuration from YAML or JSON file."""
    path = Path(config_path).expanduser().resolve()
    text = path.read_text()

    if path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    return BookConfig.model_validate(data)


def _scan_existing_knowledge() -> dict[int, list[str]]:
    """Scan knowledge directory to understand what's already been researched.

    Returns a mapping of chapter number to list of questions already covered.
    """
    knowledge: dict[int, list[str]] = {}
    knowledge_dir = settings.knowledge_dir

    if not knowledge_dir.exists():
        return knowledge

    for chapter_dir in sorted(knowledge_dir.iterdir()):
        if not chapter_dir.is_dir() or not chapter_dir.name.startswith("chapter-"):
            continue

        try:
            chapter_num = int(chapter_dir.name.split("-")[1])
        except (IndexError, ValueError):
            continue

        questions: list[str] = []
        for json_file in sorted(chapter_dir.glob("sprint-*.json")):
            try:
                data = json.loads(json_file.read_text())
                findings = SprintFindings.model_validate(data)
                questions.extend(f.question for f in findings.findings)
            except Exception:
                continue

        if questions:
            knowledge[chapter_num] = questions

    return knowledge


def _get_chapter_context(chapter_num: int) -> str:
    """Load existing research for a chapter as context for the researcher."""
    chapter_dir = settings.knowledge_dir / f"chapter-{chapter_num:02d}"
    if not chapter_dir.exists():
        return ""

    context_parts: list[str] = []
    for md_file in sorted(chapter_dir.glob("sprint-*.md")):
        try:
            content = md_file.read_text()
            context_parts.append(content)
        except Exception:
            continue

    full_context = "\n\n".join(context_parts)
    # Truncate to avoid overwhelming the researcher's context
    max_chars = settings.max_findings_tokens * 4
    if len(full_context) > max_chars:
        full_context = full_context[:max_chars] + "\n\n[... earlier research truncated ...]"

    return full_context


def _count_existing_sprints() -> int:
    """Count how many sprints have already been run."""
    sprints_dir = settings.sprints_dir
    if not sprints_dir.exists():
        return 0
    return len(list(sprints_dir.glob("sprint-[0-9]*.json")))


def run(config_path: str, *, max_sprints: int | None = None, dry_run: bool = False) -> None:
    """Run the research sprint cycle."""
    # 1. Load book config
    book_config = _load_book_config(config_path)
    print(f"Book: {book_config.title}")
    print(f"Chapters: {len(book_config.chapters)}")
    print()

    # Ensure project directories exist
    settings.project_dir.mkdir(parents=True, exist_ok=True)
    settings.sprints_dir.mkdir(parents=True, exist_ok=True)
    settings.knowledge_dir.mkdir(parents=True, exist_ok=True)

    # 2. Scan existing knowledge
    existing_knowledge = _scan_existing_knowledge()
    if existing_knowledge:
        print("Existing research coverage:")
        for ch_num, questions in sorted(existing_knowledge.items()):
            print(f"  Chapter {ch_num}: {len(questions)} questions covered")
        print()
    else:
        print("No existing research found. Starting fresh.")
        print()

    sprint_limit = max_sprints if max_sprints is not None else settings.max_sprints_per_run
    sprint_offset = _count_existing_sprints()
    follow_up_feedback: str | None = None

    # 3. Sprint cycle
    for i in range(sprint_limit):
        sprint_number = sprint_offset + i + 1
        print(f"{'=' * 60}")
        print(f"SPRINT {sprint_number}")
        print(f"{'=' * 60}")
        print()

        # a. Plan
        print("--- Planning ---")
        contract = create_sprint(
            book_config,
            existing_knowledge,
            sprint_number,
            follow_up_feedback=follow_up_feedback,
        )
        print(f"  Target: Chapter {contract.chapter}")
        print(f"  Questions: {len(contract.questions)}")
        print(f"  Priority: {contract.priority}")
        print()

        if dry_run:
            print("  [DRY RUN] Skipping research and verification.")
            print()
            follow_up_feedback = None
            continue

        # b. Research
        print("--- Researching ---")
        chapter_context = _get_chapter_context(contract.chapter)
        findings = execute_sprint(contract, chapter_context=chapter_context)
        print(f"  Collected {len(findings.findings)} findings")
        print()

        # c. Verify
        print("--- Verifying ---")
        result = verify_sprint(contract, findings)
        print(render_verification(result))
        print()

        # d/e. Decide next action
        if result.passed:
            print(f"  PASSED (score: {result.scores.overall}/10)")
            follow_up_feedback = None
            # Update knowledge index
            existing_knowledge = _scan_existing_knowledge()
        else:
            print(
                f"  FAILED (score: {result.scores.overall}/10, "
                f"threshold: {settings.score_threshold})"
            )
            follow_up_feedback = (
                f"Previous sprint {contract.sprint_id} scored {result.scores.overall}/10.\n"
                f"Feedback: {result.feedback}\n"
                f"Follow-up questions: {', '.join(result.follow_up_questions)}"
            )

        print()

    # 4. Final summary
    print(f"{'=' * 60}")
    print("RESEARCH SUMMARY")
    print(f"{'=' * 60}")
    print()
    summary = render_knowledge_summary(book_config, settings.knowledge_dir)
    print(summary)


def print_summary(config_path: str) -> None:
    """Print current knowledge summary without running sprints."""
    book_config = _load_book_config(config_path)
    summary = render_knowledge_summary(book_config, settings.knowledge_dir)
    print(summary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Book research harness using generator-evaluator sprint cycles"
    )
    parser.add_argument("config", help="Path to book config YAML/JSON file")
    parser.add_argument(
        "--max-sprints",
        type=int,
        default=None,
        help=f"Max research sprints to run (default: {settings.max_sprints_per_run})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan sprints without executing research or verification",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Just print current knowledge summary and exit",
    )
    args = parser.parse_args(argv)

    if args.summary:
        print_summary(args.config)
    else:
        run(args.config, max_sprints=args.max_sprints, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
