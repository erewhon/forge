from __future__ import annotations

from agents.book_researcher.config import settings
from agents.book_researcher.models import BookConfig, SprintContract
from agents.shared.llm import complete, extract_json

_SYSTEM_PROMPT = """\
You are a research planner for a non-fiction book. Given the book outline and \
existing research coverage, identify the most important gap and create a focused \
research sprint. Each sprint should investigate 2-4 specific questions.

Return ONLY valid JSON with these fields:
{
  "sprint_id": "<will be overridden>",
  "chapter": <chapter number>,
  "questions": ["question 1", "question 2", ...],
  "success_criteria": ["criterion 1", "criterion 2", ...],
  "priority": "high" | "medium" | "low"
}
"""


def create_sprint(
    book_config: BookConfig,
    existing_knowledge: dict[int, list[str]],
    sprint_number: int,
    follow_up_feedback: str | None = None,
) -> SprintContract:
    """Create a sprint contract identifying the next research priority."""
    sprint_id = f"{sprint_number:03d}"

    # Build chapter summary
    chapters_info = []
    for ch in book_config.chapters:
        covered = existing_knowledge.get(ch.number, [])
        coverage_note = f" (already researched: {', '.join(covered)})" if covered else " (no research yet)"
        chapters_info.append(
            f"  Chapter {ch.number}: {ch.title} - {ch.description}{coverage_note}\n"
            f"    Open questions: {', '.join(ch.research_questions)}"
        )

    user_msg = (
        f"Book: {book_config.title}\n"
        f"Description: {book_config.description}\n\n"
        f"Chapters:\n" + "\n".join(chapters_info)
    )

    if follow_up_feedback:
        user_msg += (
            f"\n\nIMPORTANT - Previous sprint did not pass verification. "
            f"Feedback from verifier:\n{follow_up_feedback}\n"
            f"Create a follow-up sprint addressing these gaps."
        )

    user_msg += f"\n\nThis is sprint #{sprint_number}. Create the next research sprint."

    print(f"  Planning sprint {sprint_id}...")
    try:
        response_text = complete(
            settings.llm_cfg(),
            system=_SYSTEM_PROMPT,
            user_message=user_msg,
            model=settings.synthesis_model,
        )
        data = extract_json(response_text)
    except Exception as e:
        print(f"  WARNING: LLM call failed for planner: {e}")
        # Fallback: pick the first chapter with no coverage
        fallback_ch = book_config.chapters[0]
        for ch in book_config.chapters:
            if ch.number not in existing_knowledge:
                fallback_ch = ch
                break
        data = {
            "chapter": fallback_ch.number,
            "questions": fallback_ch.research_questions[:3],
            "success_criteria": ["Provide substantive answers to all questions"],
            "priority": "high",
        }

    contract = SprintContract(
        sprint_id=sprint_id,
        chapter=data.get("chapter", book_config.chapters[0].number),
        questions=data.get("questions", []),
        success_criteria=data.get("success_criteria", []),
        priority=data.get("priority", "medium"),
    )

    # Write contract to disk
    sprints_dir = settings.sprints_dir
    sprints_dir.mkdir(parents=True, exist_ok=True)
    contract_path = sprints_dir / f"sprint-{sprint_id}.json"
    contract_path.write_text(contract.model_dump_json(indent=2))
    print(f"  Sprint contract written to {contract_path}")

    return contract
