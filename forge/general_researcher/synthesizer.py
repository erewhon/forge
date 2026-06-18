"""Synthesizer ensemble (research panel followup #2).

The lone synthesizer became a *judge-pick + graft* ensemble: generate a candidate final answer from
each of several diverse router models, let a judge pick the most coherent one, then graft in the
unique ``key_sources`` and ``open_questions`` the runners-up surfaced. The output keeps one model's
clean narrative voice (no re-blended prose) but loses none of the panel's coverage. Falls back to a
single-model synthesis when no candidate parses, so a run always produces an answer. ``synthesize``
keeps its signature and ``Synthesis`` output, so the main loop is unchanged.

Every model boundary goes through the harness's ``structured()`` primitive, so candidates, the
judge's verdict, and the fallback are all *validated Pydantic models* — no hand-rolled JSON parsing.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from agents.general_researcher.config import settings
from agents.general_researcher.models import (
    SprintFindings,
    Synthesis,
    TopicConfig,
    VerificationResult,
)
from agents.shared.ensemble import Pool
from agents.shared.panel import build_router_executors, run_panel, structured

_JUDGE_CANDIDATE_CHARS = 6000  # bound each candidate answer in the judge's context


class _Candidate(BaseModel):
    """One model's structured synthesis — the shape every panel member and the fallback return."""

    answer: str = ""
    key_sources: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "low"
    open_questions: list[str] = Field(default_factory=list)

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, v: object) -> str:
        s = str(v).strip().lower()
        return s if s in ("high", "medium", "low") else "low"

    @field_validator("key_sources", "open_questions", mode="before")
    @classmethod
    def _as_str_list(cls, v: object) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(x) for x in v]
        return []


class _Verdict(BaseModel):
    """The judge's pick among the candidates."""

    winner: int
    rationale: str = ""


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

_JUDGE_SYSTEM_PROMPT = """\
You are choosing the single best synthesized answer from several candidates produced by different \
models for the SAME research question. Judge on:
- factual fidelity to the findings (no fabrication, no overstated certainty),
- coherence and organization (reads as one clear piece),
- specificity preserved (dates, figures, named sources, mechanisms — not generalities),
- honest handling of disagreement and uncertainty,
- usefulness to a reader.

Penalize fabrication, vagueness, and padding. Pick the index of the strongest candidate overall.

Return ONLY valid JSON:
{ "winner": <0-based index of the best candidate>, "rationale": "one sentence on why" }
"""


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _build_user_message(
    topic: TopicConfig,
    all_findings: list[SprintFindings],
    verifications: list[VerificationResult],
) -> str:
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

    verification_summary = (
        "\n".join(
            f"- Sprint {v.sprint_id}: {v.scores.overall}/10 ({'PASSED' if v.passed else 'FAILED'})"
            for v in verifications
        )
        or "(no verifications recorded)"
    )

    incomplete = not any(v.passed for v in verifications)
    caution = (
        "Note that no sprint passed verification — be appropriately cautious about claims."
        if incomplete
        else ""
    )
    user_msg = f"Original question: {topic.question}\n"
    if topic.context:
        user_msg += f"Context: {topic.context}\n"
    user_msg += (
        f"\nVerification record:\n{verification_summary}\n"
        f"\nAll findings across {len(all_findings)} sprints:\n{findings_text}\n"
        f"\nSynthesize a final answer. {caution}"
    )
    return user_msg


def _panel_pool() -> Pool:
    """A failover pool over the synthesizer panel models — used by the judge and (single-model) the
    fallback shares the same router endpoint."""
    return Pool(
        role="synth-judge",
        executors=build_router_executors(
            settings.synthesizer_panel_models,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
        ),
    )


def _generate_candidates(user_msg: str) -> tuple[list[_Candidate], list[str]]:
    """Fan the synthesis prompt across the panel; return validated candidates + their labels."""
    executors = build_router_executors(
        settings.synthesizer_panel_models,
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
    )
    panel = run_panel(
        executors=executors,
        system=_SYSTEM_PROMPT,
        user=user_msg,
        floor=settings.synthesizer_panel_floor,
        max_tokens=8192,
    )
    candidates: list[_Candidate] = []
    labels: list[str] = []
    for resp, label in zip(panel.responses, panel.member_labels):
        try:
            candidates.append(_Candidate.model_validate(resp))
            labels.append(label)
        except ValidationError:
            continue  # a member that parsed to JSON but not to our shape is dropped
    return candidates, labels


def _build_judge_user(question: str, candidates: list[_Candidate]) -> str:
    parts = [f"Research question: {question}", ""]
    for i, c in enumerate(candidates):
        answer = c.answer.strip() or "(empty)"
        if len(answer) > _JUDGE_CANDIDATE_CHARS:
            answer = answer[:_JUDGE_CANDIDATE_CHARS] + "\n\n[... candidate truncated ...]"
        parts.append(f"## Candidate {i}\n\n{answer}\n")
    parts.append("Return the winner index and a one-sentence rationale as JSON now.")
    return "\n".join(parts)


def _judge_candidates(question: str, candidates: list[_Candidate]) -> tuple[int, str]:
    """Pick the strongest candidate via a failover judge pool. Defaults to index 0 on failure."""
    verdict = structured(
        pool=_panel_pool(),
        schema=_Verdict,
        system=_JUDGE_SYSTEM_PROMPT,
        user=_build_judge_user(question, candidates),
        max_tokens=512,
        timeout=120.0,
        predicate=lambda v: 0 <= v.winner < len(candidates),
    )
    v = verdict.value
    if v is None:
        return 0, f"judge unavailable ({verdict.error}); defaulted to candidate 0"
    return v.winner, v.rationale.strip()


def _graft(candidates: list[_Candidate], winner_idx: int) -> _Candidate:
    """Winner's prose + confidence, but key_sources / open_questions unioned across the panel
    (winner's first) so nothing a runner-up surfaced is lost."""
    winner = candidates[winner_idx]
    ordered = [winner] + [c for i, c in enumerate(candidates) if i != winner_idx]
    return _Candidate(
        answer=winner.answer.strip(),
        key_sources=_dedupe([s for c in ordered for s in c.key_sources]),
        confidence=winner.confidence,
        open_questions=_dedupe([q for c in ordered for q in c.open_questions]),
    )


def _single_synthesis(user_msg: str, question: str) -> _Candidate:
    """Fallback: one direct router call when the ensemble produced no parseable candidate."""
    print("  Synthesizer ensemble produced no candidate — falling back to single-model synthesis.")
    pool = Pool(
        role="synth-fallback",
        executors=build_router_executors(
            [settings.synthesis_model],
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
        ),
    )
    res = structured(
        pool=pool,
        schema=_Candidate,
        system=_SYSTEM_PROMPT,
        user=user_msg,
        max_tokens=8192,
    )
    if res.value is not None:
        return res.value
    print(f"  WARNING: fallback synthesis call failed: {res.error}")
    return _Candidate(
        answer="Synthesis failed. Raw findings remain in the sprint files.",
        key_sources=[],
        confidence="low",
        open_questions=[question],
    )


def synthesize(
    topic: TopicConfig,
    all_findings: list[SprintFindings],
    verifications: list[VerificationResult],
) -> Synthesis:
    sprint_count = len(all_findings)
    best_score = max((v.scores.overall for v in verifications), default=0)
    incomplete = not any(v.passed for v in verifications)

    user_msg = _build_user_message(topic, all_findings, verifications)

    print(f"  Synthesizing final answer (ensemble of {len(settings.synthesizer_panel_models)})...")
    candidates, labels = _generate_candidates(user_msg)

    if not candidates:
        final = _single_synthesis(user_msg, topic.question)
    elif len(candidates) == 1:
        print(f"  Single candidate ({labels[0]}) — no judging needed.")
        final = _graft(candidates, 0)
    else:
        winner_idx, rationale = _judge_candidates(topic.question, candidates)
        winner_label = labels[winner_idx] if winner_idx < len(labels) else f"#{winner_idx}"
        print(f"  Judge picked {winner_label} of {len(candidates)} candidates: {rationale}")
        final = _graft(candidates, winner_idx)

    return Synthesis(
        question=topic.question,
        answer=final.answer.strip() or "No synthesis produced.",
        key_sources=final.key_sources,
        confidence=final.confidence,
        open_questions=final.open_questions,
        sprint_count=sprint_count,
        best_score=best_score,
        incomplete=incomplete,
    )
