from __future__ import annotations

from agents.general_researcher.config import settings
from agents.general_researcher.models import SprintContract, TopicConfig
from agents.shared.llm import complete, extract_json

_SYSTEM_PROMPT = """\
You are a research planner investigating a focused topic. Each sprint has \
2-4 specific questions plus success criteria the researcher should meet. \
Sprint 1 should attack the main question directly (or use the user's \
sub-questions if provided). Later sprints should address gaps the verifier \
flagged or deepen the weakest aspects of prior findings.

Return ONLY valid JSON:
{
  "sprint_id": "<will be overridden>",
  "questions": ["question 1", "question 2", ...],
  "success_criteria": ["criterion 1", "criterion 2", ...],
  "rationale": "why these questions, given prior findings and feedback"
}
"""


def create_sprint(
    topic: TopicConfig,
    existing_findings_summary: str,
    sprint_number: int,
    follow_up_feedback: str | None = None,
) -> SprintContract:
    sprint_id = f"{sprint_number:03d}"

    user_msg = f"Main question: {topic.question}\n"
    if topic.context:
        user_msg += f"\nContext: {topic.context}\n"

    if topic.sub_questions and sprint_number == 1:
        user_msg += (
            "\nUser-provided sub-questions:\n"
            + "\n".join(f"- {q}" for q in topic.sub_questions)
            + "\n"
        )

    if existing_findings_summary:
        user_msg += f"\nPrior research:\n{existing_findings_summary}\n"
    else:
        user_msg += "\nNo prior research yet.\n"

    if follow_up_feedback:
        user_msg += (
            f"\nVerifier feedback from previous sprint:\n{follow_up_feedback}\n"
            f"Address these gaps in the new sprint.\n"
        )

    user_msg += f"\nThis is sprint #{sprint_number}. Plan the next research sprint."

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
        # Fallback: investigate the main question directly
        data = {
            "questions": topic.sub_questions or [topic.question],
            "success_criteria": ["Provide substantive answers to all questions"],
            "rationale": f"Planner LLM call failed; using fallback. Error: {e}",
        }

    return SprintContract(
        sprint_id=sprint_id,
        questions=data.get("questions") or [topic.question],
        success_criteria=data.get("success_criteria", []),
        rationale=data.get("rationale", ""),
    )
