from __future__ import annotations

from forge.general_researcher.models import (
    SprintFindings,
    Synthesis,
    TopicConfig,
    VerificationResult,
)
from forge.shared.llm import RESEARCH_FAILED_PREFIX


def render_sprint_findings(findings: SprintFindings) -> str:
    lines = [f"# Sprint {findings.sprint_id}", ""]
    for f in findings.findings:
        lines.extend(
            [
                f"## {f.question}",
                "",
                f.answer,
                "",
                f"**Confidence:** {f.confidence}",
                "",
            ]
        )
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
        lines.append("## Follow-up Questions")
        lines.append("")
        for q in result.follow_up_questions:
            lines.append(f"- {q}")
        lines.append("")
    return "\n".join(lines)


def render_synthesis(synth: Synthesis, topic: TopicConfig) -> str:
    caveat = ""
    if synth.incomplete:
        caveat = (
            "> **Note:** No sprint reached the verification threshold. "
            f"Best score: {synth.best_score}/10. "
            "Treat conclusions as provisional.\n\n"
        )

    lines = [
        f"# {topic.question}",
        "",
    ]
    if topic.context:
        lines.extend([f"*{topic.context}*", ""])

    lines.extend(
        [
            f"*Synthesized from {synth.sprint_count} sprint(s) "
            f"(best verification score: {synth.best_score}/10, confidence: {synth.confidence})*",
            "",
            caveat + synth.answer,
            "",
        ]
    )

    if synth.key_sources:
        lines.append("## Key Sources")
        lines.append("")
        for src in synth.key_sources:
            lines.append(f"- {src}")
        lines.append("")

    if synth.open_questions:
        lines.append("## Open Questions")
        lines.append("")
        for q in synth.open_questions:
            lines.append(f"- {q}")
        lines.append("")

    return "\n".join(lines)


def render_findings_summary(all_findings: list[SprintFindings], max_chars: int) -> str:
    """Compact summary of accumulated findings, for planner prompts."""
    if not all_findings:
        return ""
    blocks = []
    for sf in all_findings:
        for f in sf.findings:
            blocks.append(
                f"- [Sprint {sf.sprint_id}] Q: {f.question}\n"
                f"  Confidence: {f.confidence}; "
                f"{len(f.sources)} source(s); "
                f"answer length: {len(f.answer)} chars"
            )
    text = "\n".join(blocks)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[... truncated ...]"
    return text


def render_findings_context(all_findings: list[SprintFindings], max_chars: int) -> str:
    """Fuller findings text for the researcher's prior-context prompt.

    Two safeguards against fabrication laundering — an early invented "fact" hardening into
    established background through repetition. Observed live: a court docket number appeared in
    sprint 1 with ZERO sources; by sprint 2 the model was restating it at high confidence with
    sources it had found for adjacent facts, and it survived four more sprints unchallenged.

    1. Unsourced and failed findings are excluded — a claim nothing backed is not prior research,
       and it is precisely the kind that gets laundered.
    2. What remains is labelled UNVERIFIED CLAIMS rather than fact, so the model re-verifies
       instead of treating the harness's own memory as a source.
    """
    if not all_findings:
        return ""
    blocks = []
    for sf in all_findings:
        for f in sf.findings:
            if f.answer.startswith(RESEARCH_FAILED_PREFIX) or not f.sources:
                continue  # never feed an unsourced or failed claim forward
            blocks.append(
                f"### Sprint {sf.sprint_id} — {f.question}\n"
                f"{f.answer}\n"
                f"Sources: {', '.join(f.sources)}\n"
            )
    if not blocks:
        return ""
    header = (
        "The following are UNVERIFIED CLAIMS from earlier sprints, not established facts. They may "
        "contain errors. Do not restate any of them as settled — if you rely on one, verify it "
        "against a source you retrieve yourself this session and cite that source. Never treat "
        "this prior-research section as itself a source.\n\n"
    )
    text = header + "\n---\n".join(blocks)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... earlier research truncated ...]"
    return text
