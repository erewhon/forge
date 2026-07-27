"""Empty / dead-sprint guards for the general research harness (Forge task 41c3a3b3).

When the tool proxy or its web egress fails, the research model returns empty content with an
HTTP 200. These tests pin the behavior that turns that silent no-op into a loud failure: an empty
response becomes an explicit failure finding (not a blank low-confidence one), a sprint with no
usable findings is detectable so the loop can abort, and the synthesizer refuses to fabricate an
answer from memory when every finding is empty.
"""

from __future__ import annotations

from forge.general_researcher import researcher, synthesizer
from forge.general_researcher.models import (
    ResearchFinding,
    SprintContract,
    SprintFindings,
    TopicConfig,
)
from forge.shared.llm import RESEARCH_FAILED_PREFIX


def _contract(questions: list[str]) -> SprintContract:
    return SprintContract(sprint_id="001", questions=questions, success_criteria=[])


def test_empty_response_becomes_failure_finding(monkeypatch):
    # whitespace-only content survives the retry and must land on the failure path
    monkeypatch.setattr(researcher, "complete_with_retry", lambda *a, **k: "   ")
    sf = researcher.execute_sprint(_contract(["Why is the sky blue?"]))
    f = sf.findings[0]
    assert f.answer.startswith(RESEARCH_FAILED_PREFIX)
    assert f.sources == []
    assert not researcher.sprint_has_content(sf)


def test_real_json_response_is_kept(monkeypatch):
    payload = (
        '{"question":"Why?","answer":"Rayleigh scattering.","sources":["s"],"confidence":"high"}'
    )
    monkeypatch.setattr(researcher, "complete_with_retry", lambda *a, **k: payload)
    sf = researcher.execute_sprint(_contract(["Why?"]))
    f = sf.findings[0]
    assert f.answer == "Rayleigh scattering."
    assert f.confidence == "high"
    assert researcher.sprint_has_content(sf)


def test_empty_answer_field_falls_back_to_prose(monkeypatch):
    # Non-empty response but the JSON's "answer" is "" -> the `or` fallback keeps the raw text.
    # This is NOT a hard failure: the model returned something, just not in the answer field.
    monkeypatch.setattr(
        researcher, "complete_with_retry", lambda *a, **k: '{"answer": "", "sources": []}'
    )
    sf = researcher.execute_sprint(_contract(["Why?"]))
    assert sf.findings[0].answer  # non-empty (prose fallback)
    assert researcher.sprint_has_content(sf)


def test_sprint_has_content_mixed_keeps_partial():
    mixed = SprintFindings(
        sprint_id="001",
        findings=[
            ResearchFinding(question="q1", answer="a real answer", sources=[], confidence="low"),
            ResearchFinding(
                question="q2",
                answer=f"{RESEARCH_FAILED_PREFIX} empty response",
                sources=[],
                confidence="low",
            ),
        ],
    )
    assert researcher.sprint_has_content(mixed)  # one good finding is enough


def test_sprint_has_content_all_empty_is_dead():
    dead = SprintFindings(
        sprint_id="002",
        findings=[
            ResearchFinding(
                question="q1",
                answer=f"{RESEARCH_FAILED_PREFIX} empty response",
                sources=[],
                confidence="low",
            ),
            ResearchFinding(question="q2", answer="   ", sources=[], confidence="low"),
        ],
    )
    assert not researcher.sprint_has_content(dead)


def test_synthesis_refuses_when_all_findings_empty(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("synthesizer must not call a model when every finding is empty")

    monkeypatch.setattr(synthesizer, "_generate_candidates", _boom)

    topic = TopicConfig(question="Why is the sky blue?")
    dead = [
        SprintFindings(
            sprint_id="001",
            findings=[
                ResearchFinding(
                    question="q",
                    answer=f"{RESEARCH_FAILED_PREFIX} empty response",
                    sources=[],
                    confidence="low",
                )
            ],
        )
    ]
    synth = synthesizer.synthesize(topic, dead, [])
    assert synth.answer.startswith(RESEARCH_FAILED_PREFIX)
    assert synth.incomplete
    assert synth.confidence == "low"
    assert synth.open_questions == ["Why is the sky blue?"]


def test_tool_failure_diagnostic_becomes_failure_finding(monkeypatch):
    """A proxy tool-failure diagnostic is non-empty, so it must be caught by its own check."""
    diag = (
        "(max tool rounds reached) — 11 of 12 tool call(s) failed. "
        "First failure — fetch_url: HTTP 403"
    )
    monkeypatch.setattr(researcher, "complete_with_retry", lambda *a, **k: diag)
    sf = researcher.execute_sprint(_contract(["What is the docket status?"]))
    f = sf.findings[0]
    assert f.answer.startswith(RESEARCH_FAILED_PREFIX)
    assert f.sources == []
    assert not researcher.sprint_has_content(sf)
