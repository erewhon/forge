"""Adversarial verification panel (ensemble harness consumer #3).

The lone verifier became a *perspective-diverse* panel: each member gets a distinct lens (source
quality, claim verification, counter-narrative, depth, actionability) so the union of their
challenges covers orthogonal failure modes rather than redundantly flagging the same obvious gap.
Every lens still scores all five dimensions, so scores stay median-aggregated across the panel
(robust to a single lenient or harsh model). The union of challenges drives the next sprint's plan.
``verify_sprint`` keeps its signature and ``VerificationResult`` output, so the
plan→research→verify→synthesize loop is unchanged.
"""

from __future__ import annotations

import math
import statistics

from forge.general_researcher.config import settings
from forge.general_researcher.models import (
    SprintContract,
    SprintFindings,
    TopicConfig,
    VerificationResult,
    VerificationScores,
)
from forge.shared.panel import build_lens_members, run_member_panel

_DIMENSIONS = (
    "source_diversity",
    "claim_verification",
    "counter_narrative",
    "depth",
    "actionability",
)
_MAX_CHALLENGES = 12

_SYSTEM_PROMPT = """\
You are an ADVERSARIAL research verifier on a panel of independent reviewers. Do NOT rubber-stamp \
— actively try to break the findings. Hunt for unsupported or overstated claims, missing \
counter-evidence, cherry-picked or single-sourced assertions, vague "studies show" attributions, \
and gaps the research glosses over. Assume a motivated skeptic is reading.

Score each dimension 1-10 (be rigorous; 7+ means genuinely solid, 5 is mediocre, below 5 has \
substantial problems):
- **source_diversity**: Multiple perspectives and source types (primary, journalism, academic, \
official)?
- **claim_verification**: Claims well-sourced and hedged when uncertain? Or uncited / vague?
- **counter_narrative**: Opposing/alternative interpretations addressed? Biases identified?
- **depth**: Specific dates, figures, named sources, mechanisms — not generalities?
- **actionability**: Enough for a reader to use the answer, or do they still need to research?

Then list CHALLENGES: specific, concrete refutations or gaps a follow-up sprint must address — \
each one actionable (what is wrong / unsupported / missing, and what to verify).

Return ONLY valid JSON:
{
  "source_diversity": <1-10>,
  "claim_verification": <1-10>,
  "counter_narrative": <1-10>,
  "depth": <1-10>,
  "actionability": <1-10>,
  "challenges": ["specific challenge 1", "..."],
  "feedback": "one-paragraph adversarial assessment",
  "follow_up_questions": ["question to close a gap", "..."]
}
"""

# Perspective-diverse lenses: each panel member hunts ONE failure mode hardest (while still scoring
# all five dimensions, so the median aggregation stays robust). The union of their challenges then
# covers orthogonal gaps instead of every model redundantly flagging the same obvious one.
_LENSES: tuple[tuple[str, str], ...] = (
    (
        "source-quality",
        "YOUR LENS: SOURCE QUALITY. Scrutinise sourcing above all — are claims backed by diverse, "
        "credible, primary or authoritative sources, or do they lean on a single source, vague "
        "'studies show' attributions, or nothing at all? Aim your challenges at sourcing gaps.",
    ),
    (
        "claim-verification",
        "YOUR LENS: CLAIM VERIFICATION. Treat every factual assertion as guilty until sourced — "
        "flag overstated, uncited, or internally inconsistent claims, and anywhere confidence "
        "outruns the evidence. Aim your challenges at unverified claims.",
    ),
    (
        "counter-narrative",
        "YOUR LENS: COUNTER-NARRATIVE DEVIL'S ADVOCATE. Argue the other side — what opposing "
        "interpretation, disconfirming evidence, or selection bias does the research ignore? Aim "
        "your challenges at the missing counter-case.",
    ),
    (
        "depth",
        "YOUR LENS: DEPTH AND SPECIFICITY. Reward concrete dates, figures, named actors, and "
        "mechanisms; punish generalities and hand-waving. Aim your challenges at where the "
        "research stays shallow.",
    ),
    (
        "actionability",
        "YOUR LENS: ACTIONABILITY. Judge whether a reader could actually use this answer as-is, or "
        "whether key questions remain open. Aim your challenges at what still blocks use.",
    ),
)


def _compute_overall(scores: VerificationScores) -> int:
    """Weighted: counter_narrative + depth at 1.5x, others at 1.0x."""
    weighted_sum = (
        scores.source_diversity * 1.0
        + scores.claim_verification * 1.0
        + scores.counter_narrative * 1.5
        + scores.depth * 1.5
        + scores.actionability * 1.0
    )
    return int(math.floor(weighted_sum / 6.0 + 0.5))


def _clamp(value: object) -> int:
    try:
        return max(1, min(10, int(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 5


def _median_dim(responses: list[dict], dim: str) -> int:
    vals = [_clamp(r.get(dim, 5)) for r in responses]
    return int(round(statistics.median(vals))) if vals else 5


def _dedupe(items: list[str], cap: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
        if len(out) >= cap:
            break
    return out


def _build_user_message(
    topic: TopicConfig, contract: SprintContract, findings: SprintFindings
) -> str:
    findings_text = []
    for f in findings.findings:
        findings_text.append(
            f"**Q: {f.question}**\n"
            f"A: {f.answer}\n"
            f"Sources: {', '.join(f.sources) if f.sources else 'None cited'}\n"
            f"Confidence: {f.confidence}\n"
        )
    findings_summary = "\n---\n".join(findings_text)
    max_chars = settings.max_findings_tokens * 4
    if len(findings_summary) > max_chars:
        findings_summary = findings_summary[:max_chars] + "\n\n[... truncated ...]"

    criteria_text = "\n".join(f"- {c}" for c in contract.success_criteria) or "(none specified)"
    msg = f"Topic: {topic.question}\n"
    if topic.context:
        msg += f"Context: {topic.context}\n"
    msg += (
        f"\nSprint {contract.sprint_id} success criteria:\n{criteria_text}\n\n"
        "Questions investigated:\n"
        + "\n".join(f"- {q}" for q in contract.questions)
        + f"\n\nFindings:\n{findings_summary}\n\n"
        "Adversarially score each dimension and list concrete challenges."
    )
    return msg


def _fallback(contract: SprintContract, reason: str) -> VerificationResult:
    scores = VerificationScores(
        source_diversity=3,
        claim_verification=3,
        counter_narrative=3,
        depth=3,
        actionability=3,
        overall=3,
    )
    return VerificationResult(
        sprint_id=contract.sprint_id,
        scores=scores,
        passed=False,
        feedback=f"Adversarial panel produced no usable verdict: {reason}",
        follow_up_questions=contract.questions,
    )


def verify_sprint(
    topic: TopicConfig,
    contract: SprintContract,
    findings: SprintFindings,
) -> VerificationResult:
    user_msg = _build_user_message(topic, contract, findings)
    threshold = topic.score_threshold or settings.score_threshold

    members = build_lens_members(
        _LENSES,
        settings.verifier_panel_models,
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        base_system=_SYSTEM_PROMPT,
    )
    print(
        f"  Verifying sprint {contract.sprint_id} "
        f"(perspective-diverse panel, {len(members)} lenses)..."
    )
    try:
        panel = run_member_panel(
            members=members,
            user=user_msg,
            floor=settings.verifier_panel_floor,
        )
    except Exception as e:  # noqa: BLE001 — verification must never crash the sprint loop
        return _fallback(contract, f"panel error: {e}")

    if not panel.responses:
        return _fallback(contract, "no member returned parseable scores")

    scores = VerificationScores(
        source_diversity=_median_dim(panel.responses, "source_diversity"),
        claim_verification=_median_dim(panel.responses, "claim_verification"),
        counter_narrative=_median_dim(panel.responses, "counter_narrative"),
        depth=_median_dim(panel.responses, "depth"),
        actionability=_median_dim(panel.responses, "actionability"),
        overall=0,
    )
    scores.overall = _compute_overall(scores)

    challenges = _dedupe(
        [c for r in panel.responses for c in (r.get("challenges") or [])], _MAX_CHALLENGES
    )
    # The next sprint must address the panel's refutations: challenges first, then any extra Qs.
    follow_ups = _dedupe(
        challenges + [q for r in panel.responses for q in (r.get("follow_up_questions") or [])],
        _MAX_CHALLENGES,
    )

    degraded = "" if panel.quorum_met else " (below floor — degraded)"
    feedback = (
        f"Perspective-diverse panel: {len(panel.responses)}/{panel.attempted} lenses{degraded}."
    )
    if challenges:
        feedback += " Top concerns: " + " | ".join(challenges[:4])

    return VerificationResult(
        sprint_id=contract.sprint_id,
        scores=scores,
        passed=scores.overall >= threshold,
        feedback=feedback,
        follow_up_questions=follow_ups,
    )
