from __future__ import annotations

from forge.general_researcher.config import settings
from forge.general_researcher.models import (
    ResearchFinding,
    SprintContract,
    SprintFindings,
)
from forge.shared.datectx import researcher_date_context
from forge.shared.llm import (
    RESEARCH_FAILED_PREFIX,
    complete_with_retry,
    extract_json,
    is_tool_failure_text,
)
from forge.shared.source_check import scrub_citations


def sprint_has_content(findings: SprintFindings) -> bool:
    """True if any finding carries real research — a non-empty answer that isn't a failure marker.

    A sprint where this is False produced nothing usable (the tool proxy/egress was down, every
    question came back empty), so the caller should abort the loop rather than run the verifier
    panel on emptiness and burn sprints.
    """
    return any(
        f.answer.strip() and not f.answer.startswith(RESEARCH_FAILED_PREFIX)
        for f in findings.findings
    )


_SYSTEM_PROMPT = """\
You are a research assistant investigating focused questions.

You have access to tools the harness's tool proxy injects automatically:
- `web_search` — DuckDuckGo search; use for general lookups
- `fetch_url` — pull the full text of a specific page

USE THESE TOOLS. Do not rely on memory for facts that may be outdated, \
specific (dates, figures, names), or contested. For every non-trivial \
claim, ground it in a source you actually retrieved this session.

ANSWER THE QUESTION ASKED, OR SAY YOU COULD NOT. If you cannot retrieve \
sources about the specific case, entity, filing, or person named in the \
question, say so plainly and set confidence to "low". Never substitute a \
different subject you happened to find — an answer about a similar-sounding \
case or a neighbouring entity is worse than no answer, because it reads as \
responsive and is not. Report retrieval failures explicitly; they are useful.

For each research question, return ONLY valid JSON:
{
  "question": "the question being answered",
  "answer": "detailed answer drawing on retrieved sources, 200-600 words",
  "sources": ["title — author/site (URL)", "..."],
  "confidence": "high" | "medium" | "low"
}

Confidence guidance:
- high: multiple independent sources retrieved, all consistent
- medium: one or two sources retrieved, or sources partially conflict
- low: search failed, sources unavailable, or topic too niche to verify
"""


def execute_sprint(
    contract: SprintContract,
    prior_context: str = "",
) -> SprintFindings:
    findings: list[ResearchFinding] = []
    raw_notes: list[str] = []

    for i, question in enumerate(contract.questions, 1):
        print(f"    Researching question {i}/{len(contract.questions)}: {question[:80]}...")

        user_msg = f"Research question: {question}\n"
        if prior_context:
            user_msg += f"\nPrior research on this topic:\n{prior_context}\n"
        user_msg += (
            "\nUse the search and fetch_url tools to gather current sources, "
            "then synthesize a thorough answer with citations."
        )

        try:
            response_text = complete_with_retry(
                settings.llm_cfg(),
                system=f"{researcher_date_context()}\n\n{_SYSTEM_PROMPT}",
                user_message=user_msg,
                model=settings.research_model,
                max_tokens=8192,
            )
            if is_tool_failure_text(response_text):
                # The proxy reported *why* its web tools failed, but that diagnostic arrives in
                # `content`, so it is non-empty and slips past the empty-content guard below —
                # landing in the findings as a real low-confidence answer. Record it as a failure.
                print(f"    WARNING: tool failure: {response_text.strip()[:120]}")
                finding = ResearchFinding(
                    question=question,
                    answer=f"{RESEARCH_FAILED_PREFIX} {response_text.strip()[:300]}",
                    sources=[],
                    confidence="low",
                )
                raw_notes.append(f"--- Question: {question} ---\n{response_text}\n")
            elif not response_text.strip():
                # Empty content (even after a retry) means the tool proxy / web egress silently
                # failed — a successful HTTP 200 with nothing in it. Record it as an explicit
                # failure, not a blank low-confidence finding, so the sprint loop can see a dead
                # sprint and abort instead of scoring emptiness. See Forge task 41c3a3b3.
                reason = "empty response (tool proxy/egress likely failed)"
                print(f"    WARNING: {reason}")
                finding = ResearchFinding(
                    question=question,
                    answer=f"{RESEARCH_FAILED_PREFIX} {reason}",
                    sources=[],
                    confidence="low",
                )
                raw_notes.append(f"--- Question: {question} ---\n[EMPTY RESPONSE]\n")
            else:
                raw_notes.append(f"--- Question: {question} ---\n{response_text}\n")
                data = extract_json(response_text)
                answer = data.get("answer") or response_text[:2500]
                sources = data.get("sources", [])
                conf = data.get("confidence", "low")
                if settings.check_sources and sources:
                    answer, sources, conf, dead = scrub_citations(
                        answer,
                        sources,
                        conf,
                        timeout=settings.check_sources_timeout,
                        proxy=settings.check_sources_proxy,
                    )
                    if dead:
                        print(
                            f"    WARNING: {len(dead)} fabricated citation(s) removed: "
                            + "; ".join(v.url or "?" for v in dead)
                        )
                finding = ResearchFinding(
                    question=data.get("question", question),
                    answer=answer,
                    sources=sources,
                    confidence=conf,
                )
        except Exception as e:
            print(f"    WARNING: LLM call failed for question: {e}")
            finding = ResearchFinding(
                question=question,
                answer=f"{RESEARCH_FAILED_PREFIX} {e}",
                sources=[],
                confidence="low",
            )
            raw_notes.append(f"--- Question: {question} ---\nERROR: {e}\n")

        findings.append(finding)

    return SprintFindings(
        sprint_id=contract.sprint_id,
        findings=findings,
        raw_search_notes="\n".join(raw_notes),
    )
