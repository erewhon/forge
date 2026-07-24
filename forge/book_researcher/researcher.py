from __future__ import annotations

from forge.book_researcher.config import settings
from forge.book_researcher.models import ResearchFinding, SprintContract, SprintFindings
from forge.book_researcher.renderer import render_sprint_findings
from forge.shared.llm import RESEARCH_FAILED_PREFIX, complete_with_retry, extract_json


def sprint_has_content(findings: SprintFindings) -> bool:
    """True if any finding carries real research — a non-empty answer that isn't a failure marker.

    A sprint where this is False produced nothing usable (tool proxy/egress down, every question
    empty), so the caller should abort rather than run the verifier panel on emptiness.
    """
    return any(
        f.answer.strip() and not f.answer.startswith(RESEARCH_FAILED_PREFIX)
        for f in findings.findings
    )


_SYSTEM_PROMPT = """\
You are a research assistant investigating topics for a non-fiction book.

You have access to tools the harness's tool proxy injects automatically:
- `web_search` — DuckDuckGo search; use for general lookups
- `tavily_search` — higher-quality AI-summarized search; prefer this for \
named entities, recent events, or specific facts
- `fetch_url` — pull the full text of a primary source

USE THESE TOOLS. Do not rely on memory for facts that may be outdated, \
specific (dates, figures, names), or contested. For every non-trivial \
claim, ground it in a source you actually retrieved this session. Prefer \
primary sources, seminal works, and named authors over generic summaries.

For each question, return ONLY valid JSON:
{
  "question": "the question being answered",
  "answer": "detailed answer drawing on retrieved sources, 200-500 words",
  "sources": ["title — author/site (URL)", "..."],
  "confidence": "high" | "medium" | "low"
}
"""


def execute_sprint(contract: SprintContract, chapter_context: str = "") -> SprintFindings:
    """Execute a research sprint by querying the LLM for each question."""
    findings: list[ResearchFinding] = []
    raw_notes: list[str] = []

    for i, question in enumerate(contract.questions, 1):
        print(f"    Researching question {i}/{len(contract.questions)}: {question[:80]}...")

        user_msg = (
            f"Research question: {question}\n\n"
            f"This is for Chapter {contract.chapter} of a non-fiction book.\n"
        )
        if chapter_context:
            user_msg += f"\nExisting research context for this chapter:\n{chapter_context}\n"
        user_msg += (
            "\nProvide a thorough, detailed answer. Cite specific sources "
            "(books, papers, articles with authors). Flag areas needing verification."
        )

        try:
            response_text = complete_with_retry(
                settings.llm_cfg(),
                system=_SYSTEM_PROMPT,
                user_message=user_msg,
                model=settings.research_model,
            )
            if not response_text.strip():
                # Empty content (even after a retry) means the tool proxy / web egress silently
                # failed. Record it as an explicit failure so the sprint loop can abort a dead
                # sprint instead of scoring emptiness. See Forge task 41c3a3b3.
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
                finding = ResearchFinding(
                    question=data.get("question", question),
                    # `or` (not dict default) so an empty "answer" string also falls back to prose
                    answer=data.get("answer") or response_text[:2000],
                    sources=data.get("sources", []),
                    confidence=data.get("confidence", "low"),
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

    sprint_findings = SprintFindings(
        sprint_id=contract.sprint_id,
        chapter=contract.chapter,
        findings=findings,
        raw_search_notes="\n".join(raw_notes),
    )

    # Write findings to knowledge directory
    chapter_dir = settings.knowledge_dir / f"chapter-{contract.chapter:02d}"
    chapter_dir.mkdir(parents=True, exist_ok=True)

    # Structured JSON
    json_path = chapter_dir / f"sprint-{contract.sprint_id}.json"
    json_path.write_text(sprint_findings.model_dump_json(indent=2))

    # Readable markdown
    md_path = chapter_dir / f"sprint-{contract.sprint_id}.md"
    md_path.write_text(render_sprint_findings(sprint_findings))

    print(f"    Findings written to {chapter_dir}/")

    return sprint_findings
