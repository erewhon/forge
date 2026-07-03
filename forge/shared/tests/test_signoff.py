"""Tests for the shared full-quorum sign-off gate (panel mocked, no LLM)."""

from __future__ import annotations

import pytest

from agents.shared import signoff as so
from agents.shared.panel import PanelResult
from agents.shared.signoff import SignoffSeat, full_quorum_signoff


def _seats(n: int) -> list[SignoffSeat]:
    # Executors are never touched when run_panel is mocked; a sentinel object suffices.
    return [SignoffSeat(provider=f"prov{i}", executor=object()) for i in range(n)]


def _panel(responses: list[dict], attempted: int) -> PanelResult:
    return PanelResult(
        responses=responses,
        member_labels=[f"m{i}" for i in range(len(responses))],
        attempted=attempted,
        quorum_met=len(responses) >= attempted,
    )


@pytest.fixture
def panel_spy(monkeypatch):
    """Patch run_panel to a canned result and record the call kwargs."""
    calls: list[dict] = []

    def install(result: PanelResult):
        def fake(**kwargs) -> PanelResult:
            calls.append(kwargs)
            return result

        monkeypatch.setattr(so, "run_panel", fake)
        return calls

    return install


def test_unanimous_approval_passes(panel_spy):
    panel_spy(_panel([{"approve": True}, {"approve": True}], attempted=2))
    r = full_quorum_signoff("DIFF", seats=_seats(2), system="SYS", ref="epic/x")
    assert r.approved
    assert r.approvals == 2
    assert r.attempted == 2
    assert r.reason == ""
    assert r.providers == ["prov0", "prov1"]


def test_one_dissent_fails_closed(panel_spy):
    panel_spy(
        _panel(
            [{"approve": True}, {"approve": False, "blockers": ["flaky sleep"]}],
            attempted=2,
        )
    )
    r = full_quorum_signoff("DIFF", seats=_seats(2), system="SYS", ref="epic/x")
    assert not r.approved
    assert r.reason == "quorum 2/2, approvals 1/2"
    assert r.blockers == ["flaky sleep"]


def test_degraded_quorum_fails_closed_even_if_responders_approve(panel_spy):
    # 3 seats, only 2 verdicts came back — both approving. Still blocked.
    panel_spy(_panel([{"approve": True}, {"approve": True}], attempted=3))
    r = full_quorum_signoff("DIFF", seats=_seats(3), system="SYS", ref="epic/x")
    assert not r.approved
    assert r.reason == "quorum 2/3, approvals 2/3"


def test_unparseable_verdict_is_not_approval(panel_spy):
    panel_spy(_panel([{"approve": True}, {"notes": "looks fine"}], attempted=2))
    r = full_quorum_signoff("DIFF", seats=_seats(2), system="SYS", ref="epic/x")
    assert not r.approved
    assert r.approvals == 1


def test_too_few_seats_fails_without_llm_call(panel_spy):
    calls = panel_spy(_panel([{"approve": True}], attempted=1))
    r = full_quorum_signoff("DIFF", seats=_seats(1), system="SYS", ref="epic/x")
    assert not r.approved
    assert "need >=2 active providers" in r.reason
    assert r.providers == ["prov0"]
    assert calls == []  # never reached the panel


def test_prompt_carries_ref_context_and_diff(panel_spy):
    calls = panel_spy(_panel([{"approve": True}, {"approve": True}], attempted=2))
    full_quorum_signoff(
        "THE-DIFF",
        seats=_seats(2),
        system="SYS",
        ref="auto-tests/foo",
        context="This change must contain ONLY test files.",
    )
    user = calls[0]["user"]
    assert user.startswith("Change: auto-tests/foo\nThis change must contain ONLY test files.")
    assert user.endswith("\nDiff:\nTHE-DIFF")
    assert calls[0]["system"] == "SYS"
    assert calls[0]["floor"] == 2  # full quorum: floor is every seat, not a majority
