from __future__ import annotations

from agents.general_researcher.config import settings
from agents.general_researcher.models import (
    Synthesis,
    SprintFindings,
    TopicConfig,
    VerificationResult,
)
from agents.shared.llm import complete, extract_json

_SYSTEM_PROMPT = """\
You combine the findings from multiple research sprints into a single \
coherent answer to the original question. Goals:

1. Produce a clear, well-organized markdown answer that someone could \
actually use — not a recap of the research process.
2. Preserve specifics: dates, figures, named sources, mechanisms.
3. Note disagreement: where sprint findings conflict, surface the conflict \
rather than smoothing it over.
4. Cite sources inline using [1], [2], ... and list them at the end.
5. Identify open questions where the research did not reach a confident \
answer.

If verification scores are mixed or low, be honest: prefix the answer with \
a one-line caveat noting which aspects are well-supported and which are \
weakly supported.

Return ONLY valid JSON:
{
  "answer": "the synthesized markdown answer",
  "key_sources": ["primary source 1", "primary source 2", ...],
  "confidence": "high" | "medium" | "low",
  "open_questions": ["question 1", "question 2", ...]
}
"""


def synthesize(
    topic: TopicConfig,
    all_findings: list[SprintFindings],
    verifications: list[VerificationResult],
) -> Synthesis:
    sprint_count = len(all_findings)
    best_score = max((v.scores.overall for v in verifications), default=0)
    incomplete = not any(v.passed for v in verifications)

    findings_blocks: list[str] = []
    for sf in all_findings:
        for f in sf.findings:
            findings_blocks.append(
                f"### Sprint {sf.sprint_id} — {f.question}\n\n"
                f"{f.answer}\n\n"
                f"Sources: {', '.join(f.sources) if f.sources else 'None cited'}\n"
                f"Confidence: {f.confidence}\n"
            )

    findings_text = "\n---\n".join(findings_blocks)
    max_chars = settings.max_findings_tokens * 6
    if len(findings_text) > max_chars:
        findings_text = findings_text[:max_chars] + "\n\n[... earlier findings truncated ...]"

    verification_summary = "\n".join(
        f"- Sprint {v.sprint_id}: {v.scores.overall}/10 "
        f"({'PASSED' if v.passed else 'FAILED'})"
        for v in verifications
    ) or "(no verifications recorded)"

    user_msg = (
        f"Original question: {topic.question}\n"
    )
    if topic.context:
        user_msg += f"Context: {topic.context}\n"
    user_msg += (
        f"\nVerification record:\n{verification_summary}\n"
        f"\nAll findings across {sprint_count} sprints:\n{findings_text}\n"
        f"\nSynthesize a final answer. {'Note that no sprint passed verification — be appropriately cautious about claims.' if incomplete else ''}"
    )

    print("  Synthesizing final answer...")
    try:
        response_text = complete(
            settings.llm_cfg(),
            system=_SYSTEM_PROMPT,
            user_message=user_msg,
            model=settings.synthesis_model,
            max_tokens=8192,
        )
        data = extract_json(response_text)
    except Exception as e:
        print(f"  WARNING: LLM call failed for synthesizer: {e}")
        data = {
            "answer": f"Synthesis failed: {e}\n\nRaw findings remain in the sprint files.",
            "key_sources": [],
            "confidence": "low",
            "open_questions": [topic.question],
        }

    return Synthesis(
        question=topic.question,
        answer=data.get("answer", "No synthesis produced."),
        key_sources=data.get("key_sources", []),
        confidence=data.get("confidence", "low"),
        open_questions=data.get("open_questions", []),
        sprint_count=sprint_count,
        best_score=best_score,
        incomplete=incomplete,
    )
