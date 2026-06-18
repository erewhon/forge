"""Tests for the synthesizer ensemble (candidates + judge mocked — no network)."""

from __future__ import annotations

from agents.general_researcher import synthesizer
from agents.general_researcher.models import (
    ResearchFinding,
    SprintFindings,
    TopicConfig,
    VerificationResult,
    VerificationScores,
)


def _topic() -> TopicConfig:
    return TopicConfig(question="Did X cause Y?")


def _findings() -> list[SprintFindings]:
    return [
        SprintFindings(
            sprint_id="001",
            findings=[
                ResearchFinding(question="q1", answer="a", sources=["s"], confidence="medium")
            ],
        )
    ]


def _verifications(passed: bool = True) -> list[VerificationResult]:
    scores = VerificationScores(
        source_diversity=8,
        claim_verification=8,
        counter_narrative=8,
        depth=8,
        actionability=8,
        overall=8,
    )
    return [
        VerificationResult(
            sprint_id="001", scores=scores, passed=passed, feedback="ok", follow_up_questions=[]
        )
    ]


def _cand(answer: str, sources: list[str], open_qs: list[str], confidence: str = "medium") -> dict:
    return {
        "answer": answer,
        "key_sources": sources,
        "confidence": confidence,
        "open_questions": open_qs,
    }


def test_judge_pick_and_graft(monkeypatch):
    candidates = [
        _cand("answer A", ["src-a", "shared"], ["q-a"], confidence="low"),
        _cand("answer B (winner)", ["src-b", "shared"], ["q-b"], confidence="high"),
        _cand("answer C", ["src-c"], ["q-c", "q-a"]),
    ]
    labels = ["router:coder", "router:qwen", "router:glm"]
    monkeypatch.setattr(synthesizer, "_generate_candidates", lambda _u: (candidates, labels))
    monkeypatch.setattr(synthesizer, "_judge_candidates", lambda _q, _c: (1, "B is most coherent"))

    synth = synthesizer.synthesize(_topic(), _findings(), _verifications())

    # winner's prose + confidence
    assert synth.answer == "answer B (winner)"
    assert synth.confidence == "high"
    # key_sources: winner's first, then unique from the runners-up (deduped 'shared')
    assert synth.key_sources == ["src-b", "shared", "src-a", "src-c"]
    # open_questions unioned, winner first, 'q-a' deduped across A and C
    assert synth.open_questions == ["q-b", "q-a", "q-c"]
    # metadata preserved
    assert synth.sprint_count == 1
    assert synth.best_score == 8
    assert not synth.incomplete


def test_single_candidate_skips_judge(monkeypatch):
    only = [_cand("solo answer", ["src"], ["q"])]
    monkeypatch.setattr(synthesizer, "_generate_candidates", lambda _u: (only, ["router:coder"]))

    def _boom(_q, _c):
        raise AssertionError("judge must not run for a single candidate")

    monkeypatch.setattr(synthesizer, "_judge_candidates", _boom)
    synth = synthesizer.synthesize(_topic(), _findings(), _verifications())
    assert synth.answer == "solo answer"
    assert synth.key_sources == ["src"]


def test_no_candidates_falls_back_to_single(monkeypatch):
    monkeypatch.setattr(synthesizer, "_generate_candidates", lambda _u: ([], []))
    monkeypatch.setattr(
        synthesizer,
        "complete",
        lambda *a, **k: (
            '{"answer": "fallback answer", "key_sources": ["f"], '
            '"confidence": "low", "open_questions": []}'
        ),
    )
    synth = synthesizer.synthesize(_topic(), _findings(), _verifications())
    assert synth.answer == "fallback answer"
    assert synth.key_sources == ["f"]


def test_fallback_hard_default_when_complete_raises(monkeypatch):
    monkeypatch.setattr(synthesizer, "_generate_candidates", lambda _u: ([], []))

    def _boom(*a, **k):
        raise RuntimeError("router down")

    monkeypatch.setattr(synthesizer, "complete", _boom)
    synth = synthesizer.synthesize(_topic(), _findings(), _verifications(passed=False))
    assert "Synthesis failed" in synth.answer
    assert synth.open_questions == ["Did X cause Y?"]  # falls back to the topic question
    assert synth.incomplete  # no sprint passed
