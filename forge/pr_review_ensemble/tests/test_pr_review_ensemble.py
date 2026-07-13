"""Unit tests for the harness-migrated PR review ensemble.

No network: reviewers are FakeExecutor-backed ReviewerSlots and the aggregator is injected, so
these pin the wiring the live smoke can't make deterministic — quorum states, skip-slot
accounting, ExecStatus->ProviderStatus mapping, and the aggregator rotation order.
"""

from __future__ import annotations

import asyncio
import json

from forge.pr_review_ensemble.aggregator import build_aggregator
from forge.pr_review_ensemble.config import settings
from forge.pr_review_ensemble.logger import log_run
from forge.pr_review_ensemble.providers import ReviewerSlot, SkipExecutor
from forge.pr_review_ensemble.renderer import render_markdown
from forge.pr_review_ensemble.runner import run_ensemble
from forge.shared.ensemble import (
    CombineResult,
    ExecResult,
    ExecStatus,
    FailureClass,
    Pool,
    Prompt,
)


class FakeExecutor:
    """Returns a scripted ExecResult regardless of the prompt."""

    def __init__(
        self,
        label: str,
        *,
        status: ExecStatus = ExecStatus.OK,
        output: str = "looks fine",
        error: str | None = None,
        failure_class: FailureClass = FailureClass.NONE,
    ) -> None:
        self.label = label
        self._result = ExecResult(
            executor=label,
            status=status,
            output=output,
            error=error,
            failure_class=failure_class,
            latency_ms=1,
        )

    async def run(self, prompt: Prompt, *, timeout: float) -> ExecResult:
        return self._result.model_copy()


def fake_slot(provider: str, model: str = "m", **kw) -> ReviewerSlot:
    ex = FakeExecutor(f"{provider}:{model}", **kw)
    return ReviewerSlot(
        provider=provider, model=model, pool=Pool(role=f"review:{provider}", executors=[ex])
    )


def skip_slot(provider: str, model: str = "m", reason: str = "disabled in config") -> ReviewerSlot:
    ex = SkipExecutor(f"{provider}:{model}", reason)
    return ReviewerSlot(
        provider=provider,
        model=model,
        pool=Pool(role=f"review:{provider}", executors=[ex]),
        skipped_reason=reason,
    )


class FakeCombiner:
    def __init__(self, *, text="SYNTH", combiner="fake:agg", used_fallback=False) -> None:
        self._result = CombineResult(text=text, combiner=combiner, used_fallback=used_fallback)
        self.called = False
        self.inputs: list[ExecResult] | None = None

    async def combine(self, inputs):
        self.called = True
        self.inputs = list(inputs)
        return self._result


def _run(slots, aggregator=None):
    return asyncio.run(
        run_ensemble(diff_text="a\nb\nc\n", pr_ref="PR-1", slots=slots, aggregator=aggregator)
    )


# --- quorum states -----------------------------------------------------------


def test_full_quorum_aggregates():
    slots = [fake_slot("anthropic"), fake_slot("local"), fake_slot("opencode_zen")]
    agg = FakeCombiner()
    res = _run(slots, agg)
    assert res.quorum_state == "full"
    assert res.providers_succeeded == ["anthropic", "local", "opencode_zen"]
    assert res.aggregated_review == "SYNTH"
    assert res.aggregator_provider == "fake:agg"
    assert agg.called and len(agg.inputs) == 3


def test_degraded_quorum_still_aggregates():
    slots = [
        fake_slot("anthropic"),
        fake_slot("local"),
        fake_slot(
            "opencode_zen",
            status=ExecStatus.ERROR,
            output="",
            error="boom",
            failure_class=FailureClass.TERMINAL,
        ),
    ]
    agg = FakeCombiner()
    res = _run(slots, agg)
    assert res.quorum_state == "degraded"
    assert res.providers_succeeded == ["anthropic", "local"]
    assert agg.called and len(agg.inputs) == 2  # only the two successes are synthesized
    zen = next(r for r in res.reviews if r.provider == "opencode_zen")
    assert zen.status == "error" and zen.error_message == "boom"


def test_failed_quorum_skips_aggregation():
    slots = [
        fake_slot("anthropic"),
        fake_slot(
            "local",
            status=ExecStatus.ERROR,
            output="",
            error="x",
            failure_class=FailureClass.TERMINAL,
        ),
        fake_slot(
            "opencode_zen",
            status=ExecStatus.TIMEOUT,
            output="",
            error="t",
            failure_class=FailureClass.TRANSIENT,
        ),
    ]
    agg = FakeCombiner()
    res = _run(slots, agg)
    assert res.quorum_state == "failed"  # 1 ok < floor of 2
    assert res.aggregated_review is None
    assert res.aggregator_provider is None
    assert not agg.called  # never synthesize a one-voice "review"


def test_skip_slot_counts_as_attempted_not_ok():
    slots = [
        fake_slot("anthropic"),
        fake_slot("local"),
        skip_slot("opencode_zen", reason="no api key configured"),
    ]
    res = _run(slots, FakeCombiner())
    assert res.quorum_state == "degraded"
    assert res.providers_attempted == ["anthropic", "local", "opencode_zen"]
    assert res.providers_succeeded == ["anthropic", "local"]
    zen = next(r for r in res.reviews if r.provider == "opencode_zen")
    assert zen.status == "skipped" and zen.error_message == "no api key configured"


def test_status_mapping():
    slots = [
        fake_slot(
            "anthropic",
            status=ExecStatus.TIMEOUT,
            output="",
            error="to",
            failure_class=FailureClass.TRANSIENT,
        ),
        fake_slot("local"),
        fake_slot(
            "opencode_zen",
            status=ExecStatus.ERROR,
            output="",
            error="er",
            failure_class=FailureClass.TERMINAL,
        ),
    ]
    res = _run(slots, FakeCombiner())
    assert {r.provider: r.status for r in res.reviews} == {
        "anthropic": "timeout",
        "local": "ok",
        "opencode_zen": "error",
    }


# --- aggregator rotation -----------------------------------------------------


def test_aggregator_rotation_excludes_inactive():
    # preferred="sonnet-5" (default) is inactive here -> rotation falls to glm then m3.
    slots = [skip_slot("sonnet-5"), fake_slot("glm"), fake_slot("m3")]
    agg = build_aggregator(slots, pr_ref="PR", n_reviews=2)
    assert [e.label for e in agg.pool.executors] == ["glm:m", "m3:m"]


def test_aggregator_promotes_configured_preferred(monkeypatch):
    monkeypatch.setattr(settings, "aggregator_provider", "m3")
    slots = [fake_slot("sonnet-5"), fake_slot("glm"), fake_slot("m3")]
    agg = build_aggregator(slots, pr_ref="PR", n_reviews=3)
    labels = [e.label for e in agg.pool.executors]
    assert labels[0] == "m3:m"
    assert set(labels) == {"m3:m", "sonnet-5:m", "glm:m"}


# --- render + log ------------------------------------------------------------


def test_render_and_log(tmp_path):
    slots = [
        fake_slot("anthropic"),
        fake_slot("local"),
        skip_slot("opencode_zen", reason="no api key configured"),
    ]
    res = _run(slots, FakeCombiner(text="SYNTH ADVISORY"))
    md = render_markdown(res)
    assert "**Reviewed by:** 2/3" in md
    assert "SYNTH ADVISORY" in md
    assert "## Raw reviews" in md

    log_path = tmp_path / "runs.jsonl"
    log_run(res, log_path=log_path)
    rec = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert rec["quorum_state"] == "degraded"
    assert rec["aggregator_used_fallback"] is False
    assert {p["provider"] for p in rec["per_provider"]} == {"anthropic", "local", "opencode_zen"}


def test_render_marks_aggregator_fallback():
    slots = [fake_slot("anthropic"), fake_slot("local")]  # 2/2 -> full
    res = _run(slots, FakeCombiner(combiner="fallback:concat", used_fallback=True, text="raw"))
    assert res.quorum_state == "full"
    md = render_markdown(res)
    assert "all synthesizers down" in md
