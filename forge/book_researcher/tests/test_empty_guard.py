"""Empty / dead-sprint guards for the book research harness (Forge task 41c3a3b3).

Mirror of the general researcher's guard tests: an empty tool-proxy response becomes an explicit
failure finding, and a sprint with no usable findings is detectable so the loop can abort. The
book researcher's ``execute_sprint`` writes findings to the knowledge dir, so these point the
project dir at a tmp path.
"""

from __future__ import annotations

from pathlib import Path

from forge.book_researcher import researcher
from forge.book_researcher.models import ResearchFinding, SprintContract, SprintFindings
from forge.shared.llm import RESEARCH_FAILED_PREFIX


def _contract(questions: list[str]) -> SprintContract:
    return SprintContract(
        sprint_id="001", chapter=1, questions=questions, success_criteria=[], priority="medium"
    )


def test_empty_response_becomes_failure_finding(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(researcher.settings, "project_dir", tmp_path)
    monkeypatch.setattr(researcher, "complete_with_retry", lambda *a, **k: "")
    sf = researcher.execute_sprint(_contract(["Who first explained it?"]))
    f = sf.findings[0]
    assert f.answer.startswith(RESEARCH_FAILED_PREFIX)
    assert f.sources == []
    assert not researcher.sprint_has_content(sf)


def test_real_response_is_kept_and_has_content(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(researcher.settings, "project_dir", tmp_path)
    payload = (
        '{"question":"Who?","answer":"Lord Rayleigh.","sources":["a book"],"confidence":"high"}'
    )
    monkeypatch.setattr(researcher, "complete_with_retry", lambda *a, **k: payload)
    sf = researcher.execute_sprint(_contract(["Who?"]))
    assert sf.findings[0].answer == "Lord Rayleigh."
    assert researcher.sprint_has_content(sf)


def test_sprint_has_content_all_empty_is_dead():
    dead = SprintFindings(
        sprint_id="002",
        chapter=1,
        findings=[
            ResearchFinding(
                question="q1",
                answer=f"{RESEARCH_FAILED_PREFIX} empty response",
                sources=[],
                confidence="low",
            ),
            ResearchFinding(question="q2", answer="  ", sources=[], confidence="low"),
        ],
    )
    assert not researcher.sprint_has_content(dead)
