"""Tests for the adversarial verification panel (run_member_panel mocked — no network)."""

from __future__ import annotations

from agents.general_researcher import verifier
from agents.general_researcher.models import (
    ResearchFinding,
    SprintContract,
    SprintFindings,
    TopicConfig,
)
from agents.shared.panel import PanelResult


def _topic() -> TopicConfig:
    return TopicConfig(question="Did X cause Y?")


def _contract() -> SprintContract:
    return SprintContract(sprint_id="001", questions=["q1"], success_criteria=["c1"])


def _findings() -> SprintFindings:
    return SprintFindings(
        sprint_id="001",
        findings=[ResearchFinding(question="q1", answer="a", sources=["s"], confidence="medium")],
    )


def _panel(responses, *, quorum=True, attempted=3) -> PanelResult:
    return PanelResult(
        responses=responses,
        member_labels=[f"m{i}" for i in range(len(responses))],
        attempted=attempted,
        quorum_met=quorum,
    )


def _score(sd, cv, cn, dp, ac, **extra) -> dict:
    d = {
        "source_diversity": sd,
        "claim_verification": cv,
        "counter_narrative": cn,
        "depth": dp,
        "actionability": ac,
    }
    d.update(extra)
    return d


def test_median_aggregation_and_pass(monkeypatch):
    resp = [
        _score(8, 8, 8, 8, 8, challenges=["c-a"], follow_up_questions=["f1"]),
        _score(9, 7, 8, 9, 7, challenges=["c-b"]),
        _score(7, 9, 8, 7, 9, challenges=["c-a"], follow_up_questions=["f2"]),
    ]
    monkeypatch.setattr(verifier, "run_member_panel", lambda **kw: _panel(resp))
    res = verifier.verify_sprint(_topic(), _contract(), _findings())

    assert res.scores.source_diversity == 8  # median(8, 9, 7)
    assert res.scores.counter_narrative == 8
    assert res.scores.overall == 8
    assert res.passed  # >= threshold 7
    # challenges deduped (c-a twice) and drive the next sprint's follow-ups
    assert res.follow_up_questions.count("c-a") == 1
    assert "c-b" in res.follow_up_questions


def test_median_is_robust_to_a_lone_lenient_grader(monkeypatch):
    # One model rubber-stamps everything 10; two skeptics give 4s. A lone grader would have passed;
    # the median (4) does not — exactly the single-model-bias the panel removes.
    resp = [
        _score(10, 10, 10, 10, 10, challenges=[]),
        _score(4, 4, 4, 4, 4, challenges=["thin sourcing"]),
        _score(4, 4, 4, 4, 4, challenges=["thin sourcing"]),
    ]
    monkeypatch.setattr(verifier, "run_member_panel", lambda **kw: _panel(resp))
    res = verifier.verify_sprint(_topic(), _contract(), _findings())
    assert res.scores.depth == 4
    assert not res.passed
    assert "thin sourcing" in res.follow_up_questions


def test_no_responses_falls_back(monkeypatch):
    monkeypatch.setattr(verifier, "run_member_panel", lambda **kw: _panel([], quorum=False))
    res = verifier.verify_sprint(_topic(), _contract(), _findings())
    assert not res.passed
    assert res.scores.overall == 3
    assert res.follow_up_questions == ["q1"]  # falls back to the contract's questions


def test_panel_error_falls_back(monkeypatch):
    def boom(**kw):
        raise RuntimeError("router down")

    monkeypatch.setattr(verifier, "run_member_panel", boom)
    res = verifier.verify_sprint(_topic(), _contract(), _findings())
    assert not res.passed
    assert "router down" in res.feedback


def test_degraded_quorum_noted(monkeypatch):
    monkeypatch.setattr(
        verifier,
        "run_member_panel",
        lambda **kw: _panel([_score(8, 8, 8, 8, 8)], quorum=False, attempted=3),
    )
    res = verifier.verify_sprint(_topic(), _contract(), _findings())
    assert "degraded" in res.feedback
