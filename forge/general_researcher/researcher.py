from __future__ import annotations

from forge.general_researcher.config import settings
from forge.general_researcher.models import (
    ResearchFinding,
    SprintContract,
    SprintFindings,
)
from forge.shared.llm import complete, extract_json

_SYSTEM_PROMPT = """\
You are a research assistant investigating focused questions.

You have access to tools the harness's tool proxy injects automatically:
- `web_search` — DuckDuckGo search; use for general lookups
- `tavily_search` — higher-quality AI-summarized search; prefer this when \
the question needs current facts, recent events, or named entities
- `fetch_url` — pull the full text of a specific page

USE THESE TOOLS. Do not rely on memory for facts that may be outdated, \
specific (dates, figures, names), or contested. For every non-trivial \
claim, ground it in a source you actually retrieved this session.

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
            response_text = complete(
                settings.llm_cfg(),
                system=_SYSTEM_PROMPT,
                user_message=user_msg,
                model=settings.research_model,
                max_tokens=8192,
            )
            raw_notes.append(f"--- Question: {question} ---\n{response_text}\n")
            data = extract_json(response_text)

            finding = ResearchFinding(
                question=data.get("question", question),
                answer=data.get("answer", response_text[:2500]),
                sources=data.get("sources", []),
                confidence=data.get("confidence", "low"),
            )
        except Exception as e:
            print(f"    WARNING: LLM call failed for question: {e}")
            finding = ResearchFinding(
                question=question,
                answer=f"Research failed: {e}",
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
