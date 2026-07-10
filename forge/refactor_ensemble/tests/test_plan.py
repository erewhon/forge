"""Tests for the refactoring ensemble (recipe mocked — no network)."""

from __future__ import annotations

import pytest

from forge.refactor_ensemble import plan as plan_mod
from forge.refactor_ensemble.models import CanonicalSmell, Verdict
from forge.refactor_ensemble.plan import _vote, collect_code, render, run_refactor
from forge.shared.panel import ItemVerdict, PanelResult
from forge.shared.recipe import RecipeResult


class _Panel:
    def __init__(self, responses):
        self.responses = responses


def test_collect_code_reads_files(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("def f():\n    return 1\n")
    ctx, files = collect_code([str(f)], max_chars=10_000)
    assert "def f()" in ctx
    assert files == [str(f)]


def test_vote_confirms_and_bumps_impact():
    smell = CanonicalSmell(id="RF-1", location="m::f", impact="low")
    panel = _Panel(
        [
            {"real": True, "confidence": "high", "impact": "high", "reasoning": "r1"},
            {"real": True, "confidence": "low", "impact": "medium", "reasoning": "r2"},
        ]
    )
    v = _vote(smell, panel)
    assert v.status == "confirmed"
    assert v.impact == "high"  # max adjusted among the real votes
    assert (v.votes_real, v.votes_total) == (2, 2)


def test_vote_rejects_unsafe_or_bikeshed():
    # both skeptics refute (behavior change / churn) → real=false → rejected
    v = _vote(CanonicalSmell(id="x", location="m"), _Panel([{"real": False, "reasoning": "churn"}]))
    assert v.status == "rejected"


def test_vote_tentative_on_split():
    panel = _Panel([{"real": True, "confidence": "low"}, {"real": False}])
    v = _vote(CanonicalSmell(id="x", location="m", impact="medium"), panel)
    assert v.status == "tentative"


def test_run_refactor_assembles_and_sorts(monkeypatch, tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    hi = CanonicalSmell(id="RF-1", location="m::big", impact="high", effort="large")
    lo = CanonicalSmell(id="RF-2", location="m::small", impact="low", effort="small")
    rej = CanonicalSmell(id="RF-3", location="m::churn", impact="medium")
    verdicts = [
        ItemVerdict(item=lo, panel=PanelResult(), verdict=Verdict("confirmed", 2, 2, "low", ["r"])),
        ItemVerdict(
            item=hi, panel=PanelResult(), verdict=Verdict("confirmed", 2, 2, "high", ["r"])
        ),
        ItemVerdict(item=rej, panel=PanelResult(), verdict=Verdict("rejected", 0, 2, "medium", [])),
    ]

    def fake_recipe(**_kwargs):
        return RecipeResult(raw=[1, 2], canonical=[hi, lo, rej], verdicts=verdicts, dedup_ok=True)

    monkeypatch.setattr(plan_mod, "discover_dedup_verify", fake_recipe)
    report = run_refactor([str(f)], "maintainability")

    assert [s.smell.id for s in report.confirmed] == ["RF-1", "RF-2"]  # impact desc
    assert [s.smell.id for s in report.rejected] == ["RF-3"]
    md = render(report)
    assert "m::big" in md
    assert "HIGH impact / large effort" in md
    assert "2 confirmed" in md


def test_run_refactor_raises_without_code(tmp_path):
    with pytest.raises(ValueError, match="no readable source"):
        run_refactor([str(tmp_path / "missing.py")], "maintainability")
