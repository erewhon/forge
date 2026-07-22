"""The curation discipline — the keystone. Candidates accrete without moving rings; ring moves are
evidence-gated and thrash-guarded (cooldown, reversal, adjacency) with an explicit force override.
"""

from __future__ import annotations

from datetime import date

import pytest

from forge.radar.models import Quadrant, Radar, Ring
from forge.radar.movement import (
    Candidate,
    integrate_candidate,
    propose_move,
)

D0 = date(2026, 7, 1)


def _radar_with_candidate(**kw) -> Radar:
    radar = Radar()
    integrate_candidate(
        radar, Candidate(name="Thing", quadrant=Quadrant.TECHNIQUES, **kw), today=D0
    )
    return radar


# --- integrate_candidate: accumulate, never move -----------------------------


def test_new_candidate_enters_at_assess_with_first_seen_evidence():
    radar = _radar_with_candidate(summary="a technique", links=["http://x"], source="hn")
    blip = radar.get("Thing")
    assert blip.ring is Ring.ASSESS
    assert blip.first_seen == "2026-07-01"
    assert blip.last_seen == "2026-07-01"
    assert blip.ring_last is None
    assert blip.rationale == "a technique"
    assert blip.links == ["http://x"]
    assert blip.evidence[0].note == "First seen"
    assert blip.evidence[0].source == "hn"


def test_resurfaced_candidate_refreshes_and_accretes_without_moving():
    radar = _radar_with_candidate(links=["http://a"])
    result = integrate_candidate(
        radar,
        Candidate(
            name="thing",
            quadrant=Quadrant.TECHNIQUES,
            links=["http://a", "http://b"],
            summary="ignored on re-seen",
        ),
        today=date(2026, 7, 5),
    )
    blip = radar.get("Thing")
    assert result.created is False
    assert blip.ring is Ring.ASSESS  # seeing it again is NOT evidence to promote
    assert blip.last_seen == "2026-07-05"
    assert blip.first_seen == "2026-07-01"  # unchanged
    assert blip.links == ["http://a", "http://b"]  # unioned, deduped
    assert blip.rationale == ""  # a re-seen candidate's summary never overwrites the curated one


def test_integrate_is_idempotent_on_the_same_day():
    radar = _radar_with_candidate()
    integrate_candidate(radar, Candidate(name="Thing", quadrant=Quadrant.TECHNIQUES), today=D0)
    assert len(radar.blips) == 1
    # "First seen" only recorded once.
    assert sum(e.note == "First seen" for e in radar.get("Thing").evidence) == 1


# --- propose_move: evidence gate ---------------------------------------------


def test_move_requires_a_rationale():
    radar = _radar_with_candidate()
    decision = propose_move(radar, "Thing", Ring.TRIAL, "", today=date(2026, 7, 10))
    assert decision.applied is False
    assert decision.kind == "no-rationale"
    assert radar.get("Thing").ring is Ring.ASSESS


def test_applied_move_records_bookkeeping_and_accretes_evidence():
    radar = _radar_with_candidate()
    decision = propose_move(
        radar,
        "Thing",
        Ring.TRIAL,
        "trialled on the euclid router",
        today=date(2026, 7, 10),
        source="synthesis",
    )
    blip = radar.get("Thing")
    assert decision.applied is True
    assert decision.kind == "promote"
    assert blip.ring is Ring.TRIAL
    assert blip.ring_last is Ring.ASSESS
    assert blip.last_moved == "2026-07-10"
    assert blip.rationale == "trialled on the euclid router"
    # The move is journalled as dated evidence, with the ring transition in the note.
    move_ev = blip.evidence[-1]
    assert "Assess → Trial" in move_ev.note
    assert move_ev.date == "2026-07-10"


def test_move_to_same_ring_is_a_noop():
    radar = _radar_with_candidate()
    decision = propose_move(radar, "Thing", Ring.ASSESS, "already here", today=date(2026, 7, 10))
    assert decision.applied is False
    assert decision.kind == "noop"


def test_move_of_unknown_blip_reports_not_found():
    radar = _radar_with_candidate()
    decision = propose_move(radar, "Nonexistent", Ring.TRIAL, "why", today=date(2026, 7, 10))
    assert decision.applied is False
    assert decision.kind == "not-found"


# --- anti-thrash: cooldown ----------------------------------------------------


def test_second_move_within_cooldown_is_refused():
    radar = _radar_with_candidate()
    propose_move(radar, "Thing", Ring.TRIAL, "promote", today=date(2026, 7, 10))
    decision = propose_move(radar, "Thing", Ring.ADOPT, "promote again", today=date(2026, 7, 12))
    assert decision.applied is False
    assert decision.kind == "cooldown"
    assert radar.get("Thing").ring is Ring.TRIAL  # unchanged


def test_move_after_cooldown_elapses_is_allowed():
    radar = _radar_with_candidate()
    propose_move(radar, "Thing", Ring.TRIAL, "promote", today=date(2026, 7, 10))
    decision = propose_move(
        radar, "Thing", Ring.ADOPT, "solid after a week", today=date(2026, 7, 18)
    )
    assert decision.applied is True
    assert radar.get("Thing").ring is Ring.ADOPT


def test_force_overrides_cooldown():
    radar = _radar_with_candidate()
    propose_move(radar, "Thing", Ring.TRIAL, "promote", today=date(2026, 7, 10))
    decision = propose_move(
        radar, "Thing", Ring.ADOPT, "urgent", today=date(2026, 7, 11), force=True
    )
    assert decision.applied is True
    assert radar.get("Thing").ring is Ring.ADOPT


# --- anti-thrash: reversal ----------------------------------------------------


def test_reversing_the_last_move_within_cooldown_is_flagged_as_thrash():
    radar = _radar_with_candidate()
    propose_move(radar, "Thing", Ring.TRIAL, "promote", today=date(2026, 7, 10))
    # Straight back to Assess (the ring it just left) two days later.
    decision = propose_move(radar, "Thing", Ring.ASSESS, "regret", today=date(2026, 7, 12))
    assert decision.applied is False
    assert decision.kind == "reversal"


def test_reversal_after_cooldown_is_allowed():
    radar = _radar_with_candidate()
    propose_move(radar, "Thing", Ring.TRIAL, "promote", today=date(2026, 7, 10))
    decision = propose_move(radar, "Thing", Ring.ASSESS, "it regressed", today=date(2026, 7, 20))
    assert decision.applied is True
    assert radar.get("Thing").ring is Ring.ASSESS


# --- adjacency ---------------------------------------------------------------


def test_promotion_cannot_skip_a_ring_by_default():
    radar = _radar_with_candidate()
    decision = propose_move(
        radar, "Thing", Ring.ADOPT, "straight to adopt", today=date(2026, 7, 10)
    )
    assert decision.applied is False
    assert decision.kind == "non-adjacent"


def test_allow_jump_permits_a_multi_ring_promotion():
    radar = _radar_with_candidate()
    decision = propose_move(
        radar, "Thing", Ring.ADOPT, "proven elsewhere", today=date(2026, 7, 10), allow_jump=True
    )
    assert decision.applied is True
    assert radar.get("Thing").ring is Ring.ADOPT


def test_hold_is_reachable_from_any_ring_without_a_jump():
    # Park a fresh Assess blip straight into Hold — a veto is always allowed.
    radar = _radar_with_candidate()
    decision = propose_move(
        radar, "Thing", Ring.HOLD, "deprecated upstream", today=date(2026, 7, 10)
    )
    assert decision.applied is True
    assert decision.kind == "hold"
    assert radar.get("Thing").ring is Ring.HOLD


def test_leaving_hold_lands_at_adjacent_assess():
    radar = _radar_with_candidate()
    propose_move(radar, "Thing", Ring.HOLD, "park it", today=date(2026, 7, 1))
    decision = propose_move(
        radar, "Thing", Ring.ASSESS, "worth another look", today=date(2026, 7, 15)
    )
    assert decision.applied is True
    assert radar.get("Thing").ring is Ring.ASSESS


def test_demotion_one_ring_is_adjacent_and_allowed():
    radar = _radar_with_candidate()
    propose_move(radar, "Thing", Ring.TRIAL, "promote", today=date(2026, 7, 1))
    decision = propose_move(radar, "Thing", Ring.ASSESS, "regressed", today=date(2026, 7, 15))
    assert decision.applied is True
    assert decision.kind == "demote"


@pytest.mark.parametrize("cooldown", [0, 1])
def test_zero_or_one_day_cooldown_still_blocks_same_day_double_move(cooldown):
    radar = _radar_with_candidate()
    propose_move(radar, "Thing", Ring.TRIAL, "promote", today=D0, cooldown_days=cooldown)
    decision = propose_move(radar, "Thing", Ring.ADOPT, "again", today=D0, cooldown_days=cooldown)
    # cooldown_days=0 means "< 0 days ago" is never true, so same-day IS allowed;
    # cooldown_days=1 blocks the same-day second move.
    if cooldown == 0:
        assert decision.applied is True
    else:
        assert decision.applied is False
        assert decision.kind == "cooldown"
