"""Tests for the book_researcher verification panel (run_member_panel mocked, no network)."""

from __future__ import annotations

import json

from forge.book_researcher import verifier
from forge.book_researcher.models import (
    ResearchFinding,
    SprintContract,
    SprintFindings,
)
from forge.shared.panel import PanelResult


def _contract() -> SprintContract:
    return SprintContract(
        sprint_id="001", chapter=1, questions=["q1"], success_criteria=["c1"], priority="high"
    )


def _findings() -> SprintFindings:
    return SprintFindings(
        sprint_id="001",
        chapter=1,
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


def _use_tmp(monkeypatch, tmp_path):
    # redirect the on-disk review write into the test's tmp dir
    monkeypatch.setattr(verifier.settings, "project_dir", tmp_path)


def test_median_aggregation_and_pass(monkeypatch, tmp_path):
    _use_tmp(monkeypatch, tmp_path)
    resp = [
        _score(8, 8, 8, 8, 8, challenges=["c-a"], follow_up_questions=["f1"]),
        _score(9, 7, 8, 9, 7, challenges=["c-b"]),
        _score(7, 9, 8, 7, 9, challenges=["c-a"]),
    ]
    monkeypatch.setattr(verifier, "run_member_panel", lambda **kw: _panel(resp))
    res = verifier.verify_sprint(_contract(), _findings())

    assert res.scores.source_diversity == 8
    assert res.scores.overall == 8
    assert res.passed
    assert res.follow_up_questions.count("c-a") == 1  # deduped
    assert "c-b" in res.follow_up_questions

    # the review was persisted to disk
    review = tmp_path / "sprints" / "sprint-001-review.json"
    assert review.is_file()
    assert json.loads(review.read_text())["passed"] is True


def test_median_robust_to_lone_lenient_grader(monkeypatch, tmp_path):
    _use_tmp(monkeypatch, tmp_path)
    resp = [
        _score(10, 10, 10, 10, 10),
        _score(4, 4, 4, 4, 4, challenges=["thin sourcing"]),
        _score(4, 4, 4, 4, 4, challenges=["thin sourcing"]),
    ]
    monkeypatch.setattr(verifier, "run_member_panel", lambda **kw: _panel(resp))
    res = verifier.verify_sprint(_contract(), _findings())
    assert res.scores.depth == 4
    assert not res.passed
    assert "thin sourcing" in res.follow_up_questions


def test_no_responses_falls_back(monkeypatch, tmp_path):
    _use_tmp(monkeypatch, tmp_path)
    monkeypatch.setattr(verifier, "run_member_panel", lambda **kw: _panel([], quorum=False))
    res = verifier.verify_sprint(_contract(), _findings())
    assert not res.passed
    assert res.scores.overall == 3
    assert res.follow_up_questions == ["q1"]


def test_panel_error_falls_back(monkeypatch, tmp_path):
    _use_tmp(monkeypatch, tmp_path)

    def boom(**kw):
        raise RuntimeError("router down")

    monkeypatch.setattr(verifier, "run_member_panel", boom)
    res = verifier.verify_sprint(_contract(), _findings())
    assert not res.passed
    assert "router down" in res.feedback
