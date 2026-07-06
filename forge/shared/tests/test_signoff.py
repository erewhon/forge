"""Tests for the shared full-quorum sign-off gate (panel mocked, no LLM)."""

from __future__ import annotations

import pytest

from agents.shared import signoff as so
from agents.shared.panel import PanelResult
from agents.shared.signoff import SignoffSeat, full_quorum_signoff


def _seats(n: int) -> list[SignoffSeat]:
    # Executors are never touched when the panel is mocked; a sentinel object suffices.
    return [SignoffSeat(provider=f"prov{i}", executor=object()) for i in range(n)]


def _panel(
    responses: list[dict],
    attempted: int,
    failures: list[tuple[str, str]] | None = None,
) -> PanelResult:
    # member_labels mirror _seats' provider names so seat verdicts map back correctly.
    return PanelResult(
        responses=responses,
        member_labels=[f"prov{i}" for i in range(len(responses))],
        attempted=attempted,
        quorum_met=len(responses) >= attempted,
        failures=failures or [],
    )


@pytest.fixture
def panel_spy(monkeypatch):
    """Patch run_member_panel to a canned result and record the call kwargs."""
    calls: list[dict] = []

    def install(result: PanelResult):
        def fake(**kwargs) -> PanelResult:
            calls.append(kwargs)
            return result

        monkeypatch.setattr(so, "run_member_panel", fake)
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
    assert [s.approve for s in r.seats] == [None]  # not attempted, not a rejection


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
    members = calls[0]["members"]
    assert [m.system for m in members] == ["SYS", "SYS"]
    assert [m.label for m in members] == ["prov0", "prov1"]  # verdicts map back to seats
    assert calls[0]["floor"] == 2  # full quorum: floor is every seat, not a majority


def test_seat_verdicts_distinguish_no_response_from_rejection(panel_spy):
    # prov0 approved, prov1 blocked, prov2 never answered (timeout) — three distinct outcomes.
    panel_spy(
        _panel(
            [{"approve": True, "notes": "clean"}, {"approve": False}],
            attempted=3,
            failures=[("prov2", "timed out after 120s")],
        )
    )
    r = full_quorum_signoff("DIFF", seats=_seats(3), system="SYS", ref="epic/x")
    assert not r.approved
    by_provider = {s.provider: s for s in r.seats}
    assert by_provider["prov0"].approve is True
    assert by_provider["prov0"].reason == "clean"
    assert by_provider["prov1"].approve is False
    assert by_provider["prov2"].approve is None
    assert "timed out" in by_provider["prov2"].reason


def test_zero_responders_carry_per_seat_reasons(panel_spy):
    # The distill-evals failure mode: quorum 0/2 because NOBODY answered. The result must say why.
    panel_spy(
        _panel(
            [],
            attempted=2,
            failures=[
                ("prov0", "AuthenticationError: missing api key"),
                ("prov1", "responded but returned no parseable JSON"),
            ],
        )
    )
    r = full_quorum_signoff("DIFF", seats=_seats(2), system="SYS", ref="epic/x")
    assert not r.approved
    assert r.reason == "quorum 0/2, approvals 0/2"
    assert [s.approve for s in r.seats] == [None, None]
    assert "missing api key" in r.seats[0].reason
    assert "no parseable JSON" in r.seats[1].reason
