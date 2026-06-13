from __future__ import annotations

import math

from agents.book_researcher.config import settings
from agents.book_researcher.models import (
    SprintContract,
    SprintFindings,
    VerificationResult,
    VerificationScores,
)
from agents.shared.llm import complete, extract_json

_SYSTEM_PROMPT = """\
You are a research quality verifier for a non-fiction book project. You evaluate \
research findings against specific criteria and provide honest, constructive scores.

Score each dimension from 1-10:
- **source_diversity**: Multiple perspectives represented? Different source types \
(books, papers, primary sources, journalism)?
- **claim_verification**: Are claims well-supported? Appropriately hedged when uncertain?
- **counter_narrative**: Are opposing viewpoints addressed? Potential biases identified?
- **depth**: Beyond surface-level? Specific details, dates, figures, named sources?
- **actionability**: Can a writer use this to draft a chapter section? Enough material?

Be rigorous. A score of 7+ means genuinely good research. 5 is mediocre. Below 5 \
means significant gaps.

Return ONLY valid JSON:
{
  "source_diversity": <1-10>,
  "claim_verification": <1-10>,
  "counter_narrative": <1-10>,
  "depth": <1-10>,
  "actionability": <1-10>,
  "feedback": "specific feedback on strengths and weaknesses",
  "follow_up_questions": ["question 1 to address gaps", "question 2", ...]
}
"""


def _compute_overall(scores: VerificationScores) -> int:
    """Compute weighted average overall score.

    counter_narrative and depth are weighted 1.5x.
    """
    weighted_sum = (
        scores.source_diversity * 1.0
        + scores.claim_verification * 1.0
        + scores.counter_narrative * 1.5
        + scores.depth * 1.5
        + scores.actionability * 1.0
    )
    total_weight = 1.0 + 1.0 + 1.5 + 1.5 + 1.0
    return int(math.floor(weighted_sum / total_weight + 0.5))


def verify_sprint(contract: SprintContract, findings: SprintFindings) -> VerificationResult:
    """Verify sprint findings against success criteria."""
    # Build findings summary, truncated to max_findings_tokens
    findings_text = []
    for f in findings.findings:
        entry = (
            f"**Q: {f.question}**\n"
            f"A: {f.answer}\n"
            f"Sources: {', '.join(f.sources) if f.sources else 'None cited'}\n"
            f"Confidence: {f.confidence}\n"
        )
        findings_text.append(entry)

    findings_summary = "\n---\n".join(findings_text)
    # Rough token estimate: ~4 chars per token
    max_chars = settings.max_findings_tokens * 4
    if len(findings_summary) > max_chars:
        findings_summary = findings_summary[:max_chars] + "\n\n[... truncated ...]"

    criteria_text = "\n".join(f"- {c}" for c in contract.success_criteria)

    user_msg = (
        f"Sprint {contract.sprint_id} for Chapter {contract.chapter}\n\n"
        f"Success criteria:\n{criteria_text}\n\n"
        f"Questions investigated:\n"
        + "\n".join(f"- {q}" for q in contract.questions)
        + f"\n\nFindings:\n{findings_summary}\n\n"
        f"Score each dimension and provide specific feedback."
    )

    print(f"  Verifying sprint {contract.sprint_id}...")
    try:
        response_text = complete(
            settings.llm_cfg(),
            system=_SYSTEM_PROMPT,
            user_message=user_msg,
            model=settings.synthesis_model,
        )
        data = extract_json(response_text)
    except Exception as e:
        print(f"  WARNING: LLM call failed for verifier: {e}")
        data = {
            "source_diversity": 3,
            "claim_verification": 3,
            "counter_narrative": 3,
            "depth": 3,
            "actionability": 3,
            "feedback": f"Verification failed due to LLM error: {e}",
            "follow_up_questions": contract.questions,
        }

    scores = VerificationScores(
        source_diversity=max(1, min(10, data.get("source_diversity", 5))),
        claim_verification=max(1, min(10, data.get("claim_verification", 5))),
        counter_narrative=max(1, min(10, data.get("counter_narrative", 5))),
        depth=max(1, min(10, data.get("depth", 5))),
        actionability=max(1, min(10, data.get("actionability", 5))),
        overall=0,  # computed below
    )
    scores.overall = _compute_overall(scores)

    passed = scores.overall >= settings.score_threshold

    result = VerificationResult(
        sprint_id=contract.sprint_id,
        scores=scores,
        passed=passed,
        feedback=data.get("feedback", "No feedback available."),
        follow_up_questions=data.get("follow_up_questions", []),
    )

    # Write review to disk
    review_path = settings.sprints_dir / f"sprint-{contract.sprint_id}-review.json"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(result.model_dump_json(indent=2))
    print(f"  Review written to {review_path}")

    return result
