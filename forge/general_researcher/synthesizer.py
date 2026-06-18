"""Synthesizer ensemble (research panel followup #2).

The lone synthesizer became a *judge-pick + graft* ensemble: generate a candidate final answer from
each of several diverse router models, let a judge pick the most coherent one, then graft in the
unique ``key_sources`` and ``open_questions`` the runners-up surfaced. The output keeps one model's
clean narrative voice (no re-blended prose) but loses none of the panel's coverage. Falls back to a
single-model synthesis when no candidate parses, so a run always produces an answer. ``synthesize``
keeps its signature and ``Synthesis`` output, so the main loop is unchanged.
"""

from __future__ import annotations

import asyncio

from agents.general_researcher.config import settings
from agents.general_researcher.models import (
    SprintFindings,
    Synthesis,
    TopicConfig,
    VerificationResult,
)
from agents.shared.ensemble import Pool, Prompt
from agents.shared.llm import complete, extract_json
from agents.shared.panel import build_router_executors, run_panel

_JUDGE_CANDIDATE_CHARS = 6000  # bound each candidate answer in the judge's context

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


def _generate_candidates(user_msg: str) -> tuple[list[dict], list[str]]:
    """Fan the synthesis prompt across the panel; return parsed candidate dicts + their labels."""
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
    return panel.responses, panel.member_labels


def _build_judge_user(question: str, candidates: list[dict]) -> str:
    parts = [f"Research question: {question}", ""]
    for i, c in enumerate(candidates):
        answer = str(c.get("answer", "")).strip() or "(empty)"
        if len(answer) > _JUDGE_CANDIDATE_CHARS:
            answer = answer[:_JUDGE_CANDIDATE_CHARS] + "\n\n[... candidate truncated ...]"
        parts.append(f"## Candidate {i}\n\n{answer}\n")
    parts.append("Return the winner index and a one-sentence rationale as JSON now.")
    return "\n".join(parts)


def _judge_candidates(question: str, candidates: list[dict]) -> tuple[int, str]:
    """Pick the strongest candidate via a failover judge pool. Defaults to index 0 on failure."""
    executors = build_router_executors(
        settings.synthesizer_panel_models,
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
    )
    pool = Pool(role="synth-judge", executors=executors)
    prompt = Prompt(
        system=_JUDGE_SYSTEM_PROMPT,
        user=_build_judge_user(question, candidates),
        max_tokens=512,
    )

    def _valid(text: str) -> bool:
        data = extract_json(text)
        winner = data.get("winner")
        try:
            return 0 <= int(winner) < len(candidates)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False

    result = asyncio.run(pool.run(prompt, timeout=120.0, validate=_valid))
    if not result.ok:
        return 0, f"judge unavailable ({result.error}); defaulted to candidate 0"
    data = extract_json(result.output)
    return int(data["winner"]), str(data.get("rationale", "")).strip()


def _graft(candidates: list[dict], winner_idx: int) -> dict:
    """Winner's prose + confidence, but key_sources / open_questions unioned across the panel
    (winner's first) so nothing a runner-up surfaced is lost."""
    winner = candidates[winner_idx]
    ordered = [winner] + [c for i, c in enumerate(candidates) if i != winner_idx]
    key_sources = _dedupe([s for c in ordered for s in (c.get("key_sources") or [])])
    open_questions = _dedupe([q for c in ordered for q in (c.get("open_questions") or [])])
    return {
        "answer": str(winner.get("answer", "")).strip() or "No synthesis produced.",
        "key_sources": key_sources,
        "confidence": winner.get("confidence", "low"),
        "open_questions": open_questions,
    }


def _single_synthesis(user_msg: str, question: str) -> dict:
    """Fallback: one direct model call when the ensemble produced no parseable candidate."""
    print("  Synthesizer ensemble produced no candidate — falling back to single-model synthesis.")
    try:
        response_text = complete(
            settings.llm_cfg(),
            system=_SYSTEM_PROMPT,
            user_message=user_msg,
            model=settings.synthesis_model,
            max_tokens=8192,
        )
        data = extract_json(response_text)
        if data:
            return data
    except Exception as e:  # noqa: BLE001 — synthesis must always return something usable
        print(f"  WARNING: fallback synthesis call failed: {e}")
    return {
        "answer": "Synthesis failed. Raw findings remain in the sprint files.",
        "key_sources": [],
        "confidence": "low",
        "open_questions": [question],
    }


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
        data = _single_synthesis(user_msg, topic.question)
    elif len(candidates) == 1:
        print(f"  Single candidate ({labels[0]}) — no judging needed.")
        data = _graft(candidates, 0)
    else:
        winner_idx, rationale = _judge_candidates(topic.question, candidates)
        winner_label = labels[winner_idx] if winner_idx < len(labels) else f"#{winner_idx}"
        print(f"  Judge picked {winner_label} of {len(candidates)} candidates: {rationale}")
        data = _graft(candidates, winner_idx)

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
