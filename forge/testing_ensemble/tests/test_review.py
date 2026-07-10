"""Tests for the testing-review ensemble (recipe mocked — no network)."""

from __future__ import annotations

import pytest

from forge.shared.panel import ItemVerdict, PanelResult
from forge.shared.recipe import RecipeResult
from forge.testing_ensemble import review as review_mod
from forge.testing_ensemble.models import CanonicalGap, Verdict
from forge.testing_ensemble.review import _vote, collect_context, render, run_review


class _Panel:
    def __init__(self, responses):
        self.responses = responses


def test_collect_context_classifies_source_and_tests(tmp_path):
    (tmp_path / "mod.py").write_text("def f():\n    return 1\n")
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    (test_dir / "test_mod.py").write_text("def test_f():\n    assert True\n")

    context, sources, tests = collect_context([str(tmp_path)])
    assert sources == [str(tmp_path / "mod.py")]
    assert tests == [str(test_dir / "test_mod.py")]
    assert "## SOURCE UNDER TEST" in context
    assert "## EXISTING TESTS" in context
    assert "def f()" in context and "def test_f()" in context


def test_collect_context_notes_missing_tests(tmp_path):
    (tmp_path / "mod.py").write_text("def f():\n    return 1\n")
    context, sources, tests = collect_context([str(tmp_path)])
    assert sources and not tests
    assert "No existing tests found" in context


def test_vote_confirms_and_bumps_severity():
    gap = CanonicalGap(id="TG-1", target="mod::f", severity="low")
    panel = _Panel(
        [
            {"real": True, "confidence": "high", "severity": "high", "reasoning": "r1"},
            {"real": True, "confidence": "low", "severity": "medium", "reasoning": "r2"},
        ]
    )
    v = _vote(gap, panel)
    assert v.status == "confirmed"
    assert v.severity == "high"
    assert (v.votes_real, v.votes_total) == (2, 2)


def test_vote_rejects_when_already_covered():
    # both skeptics say the case is already covered → real=false → rejected
    v = _vote(CanonicalGap(id="x", target="t"), _Panel([{"real": False, "reasoning": "covered"}]))
    assert v.status == "rejected"


def test_run_review_assembles_and_sorts(monkeypatch, tmp_path):
    (tmp_path / "mod.py").write_text("def f():\n    return 1\n")
    hi = CanonicalGap(id="TG-1", target="mod::big", severity="high")
    lo = CanonicalGap(id="TG-2", target="mod::small", severity="low")
    rej = CanonicalGap(id="TG-3", target="mod::covered", severity="medium")
    verdicts = [
        ItemVerdict(item=lo, panel=PanelResult(), verdict=Verdict("confirmed", 2, 2, "low", ["r"])),
        ItemVerdict(
            item=hi, panel=PanelResult(), verdict=Verdict("confirmed", 2, 2, "high", ["r"])
        ),
        ItemVerdict(item=rej, panel=PanelResult(), verdict=Verdict("rejected", 0, 2, "medium", [])),
    ]

    def fake_recipe(**_kwargs):
        return RecipeResult(raw=[1, 2], canonical=[hi, lo, rej], verdicts=verdicts, dedup_ok=True)

    monkeypatch.setattr(review_mod, "discover_dedup_verify", fake_recipe)
    report = run_review([str(tmp_path)], "coverage")

    assert [s.gap.id for s in report.confirmed] == ["TG-1", "TG-2"]  # severity desc
    assert [s.gap.id for s in report.rejected] == ["TG-3"]
    md = render(report)
    assert "mod::big" in md
    assert "2 confirmed" in md


def test_run_review_raises_without_source(tmp_path):
    with pytest.raises(ValueError, match="no readable source"):
        run_review([str(tmp_path / "missing.py")], "coverage")
