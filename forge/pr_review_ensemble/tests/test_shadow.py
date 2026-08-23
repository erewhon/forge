"""The shadow pass: both rosters run on the same diff, results rendered side by side.

No network: the roster builders are monkeypatched to FakeExecutor-backed slots (whose executors
also serve as the aggregator pool members), so these pin the orchestration — both rosters
attempted, pr_refs suffixed for the logs, and the comparison render carrying both advisories.
"""

from __future__ import annotations

import asyncio

from forge.pr_review_ensemble import shadow as shadow_mod
from forge.pr_review_ensemble.tests.test_pr_review_ensemble import fake_slot


def _run_shadow(monkeypatch) -> shadow_mod.ShadowResult:
    monkeypatch.setattr(
        shadow_mod,
        "build_reviewer_slots",
        lambda: [fake_slot("sonnet-5"), fake_slot("ling3"), fake_slot("gemma")],
    )
    monkeypatch.setattr(
        shadow_mod,
        "build_local_reviewer_slots",
        lambda: [fake_slot("ling3"), fake_slot("gemma"), fake_slot("lightning")],
    )
    return asyncio.run(shadow_mod.run_shadow(diff_text="diff --git a b", pr_ref="repo#1"))


def test_shadow_runs_both_rosters(monkeypatch):
    result = _run_shadow(monkeypatch)

    assert result.baseline.providers_attempted == ["sonnet-5", "ling3", "gemma"]
    assert result.local.providers_attempted == ["ling3", "gemma", "lightning"]
    # The two runs log under distinguishable refs.
    assert result.baseline.pr_ref == "repo#1 [baseline]"
    assert result.local.pr_ref == "repo#1 [local]"


def test_render_shadow_carries_both_advisories(monkeypatch):
    result = _run_shadow(monkeypatch)

    md = shadow_mod.render_shadow(result)

    assert md.startswith("# Shadow review comparison — repo#1\n")
    assert "## Production advisory (baseline)" in md
    assert "## All-local advisory" in md
    assert "lightning" in md  # the local trio's third family is visible in the summary table
