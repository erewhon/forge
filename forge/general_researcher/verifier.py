from __future__ import annotations

import math

from agents.general_researcher.config import settings
from agents.general_researcher.models import (
    SprintContract,
    SprintFindings,
    TopicConfig,
    VerificationResult,
    VerificationScores,
)
from agents.shared.llm import complete, extract_json

_SYSTEM_PROMPT = """\
You evaluate research findings against the topic and the sprint's success \
criteria, scoring each dimension from 1-10:

- **source_diversity**: Multiple perspectives represented? Different source \
types (primary sources, journalism, academic, official records)?
- **claim_verification**: Are claims well-sourced? Hedged when uncertain? \
Watch for claims with no citation or vague "studies show" attributions.
- **counter_narrative**: Are opposing or alternative interpretations \
addressed? Are biases identified?
- **depth**: Beyond surface-level? Specific dates, figures, named sources, \
mechanisms — not just generalities?
- **actionability**: Does this give a reader enough to actually use the \
answer, or does it leave them needing to do their own research?

Be rigorous. 7+ means genuinely good research that addresses the topic \
well. 5 is mediocre — has something but with significant gaps. Below 5 \
means substantial problems.

Return ONLY valid JSON:
{
  "source_diversity": <1-10>,
  "claim_verification": <1-10>,
  "counter_narrative": <1-10>,
  "depth": <1-10>,
  "actionability": <1-10>,
  "feedback": "specific feedback on strengths and weaknesses",
  "follow_up_questions": ["question 1 to address gaps", ...]
}
"""


def _compute_overall(scores: VerificationScores) -> int:
    """Weighted: counter_narrative + depth at 1.5x, others at 1.0x."""
    weighted_sum = (
        scores.source_diversity * 1.0
        + scores.claim_verification * 1.0
        + scores.counter_narrative * 1.5
        + scores.depth * 1.5
        + scores.actionability * 1.0
    )
    total_weight = 6.0
    return int(math.floor(weighted_sum / total_weight + 0.5))


def verify_sprint(
    topic: TopicConfig,
    contract: SprintContract,
    findings: SprintFindings,
) -> VerificationResult:
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
    max_chars = settings.max_findings_tokens * 4
    if len(findings_summary) > max_chars:
        findings_summary = findings_summary[:max_chars] + "\n\n[... truncated ...]"

    criteria_text = "\n".join(f"- {c}" for c in contract.success_criteria) or "(none specified)"
    threshold = topic.score_threshold or settings.score_threshold

    user_msg = (
        f"Topic: {topic.question}\n"
    )
    if topic.context:
        user_msg += f"Context: {topic.context}\n"
    user_msg += (
        f"\nSprint {contract.sprint_id} success criteria:\n{criteria_text}\n\n"
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
        overall=0,
    )
    scores.overall = _compute_overall(scores)

    return VerificationResult(
        sprint_id=contract.sprint_id,
        scores=scores,
        passed=scores.overall >= threshold,
        feedback=data.get("feedback", "No feedback available."),
        follow_up_questions=data.get("follow_up_questions", []),
    )
