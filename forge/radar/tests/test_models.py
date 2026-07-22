"""Radar domain types: slug identity, evidence dedup key, and the Radar container's index/upsert."""

from __future__ import annotations

from forge.radar.models import (
    RING_ORDER,
    Blip,
    Evidence,
    Quadrant,
    Radar,
    Ring,
    ring_index,
    slugify,
)


def _blip(name: str, ring: Ring = Ring.ASSESS, quadrant: Quadrant = Quadrant.TECHNIQUES) -> Blip:
    return Blip(
        name=name, quadrant=quadrant, ring=ring, first_seen="2026-07-21", last_seen="2026-07-21"
    )


def test_slugify_is_stable_across_case_punctuation_and_accents():
    assert slugify("Qwen3-Coder 30B") == "qwen3-coder-30b"
    assert slugify("qwen3 coder 30b") == "qwen3-coder-30b"
    assert slugify("  Café  Réasoning!! ") == "cafe-reasoning"
    # Different display names that normalise the same map to one identity.
    assert slugify("Structured Tool-Calling") == slugify("structured  tool  calling")


def test_ring_order_is_centre_outward():
    assert RING_ORDER == [Ring.ADOPT, Ring.TRIAL, Ring.ASSESS, Ring.HOLD]
    assert ring_index(Ring.ADOPT) == 0
    assert ring_index(Ring.HOLD) == 3
    # Promotion lowers the index.
    assert ring_index(Ring.TRIAL) < ring_index(Ring.ASSESS)


def test_evidence_key_dedups_on_date_and_note_ignoring_source():
    a = Evidence(date="2026-07-21", note="benchmarked", source="hands-on")
    b = Evidence(date="2026-07-21", note=" benchmarked ", source="different-source")
    assert a.key() == b.key()


def test_radar_upsert_replaces_in_place_preserving_order():
    radar = Radar(blips=[_blip("Alpha"), _blip("Beta"), _blip("Gamma")])
    updated = _blip("beta", ring=Ring.TRIAL)  # same slug, different display case
    radar.upsert(updated)

    assert [b.name for b in radar.blips] == [
        "Alpha",
        "beta",
        "Gamma",
    ]  # position kept, name replaced
    assert radar.get("Beta").ring is Ring.TRIAL


def test_radar_upsert_appends_new_blip():
    radar = Radar(blips=[_blip("Alpha")])
    radar.upsert(_blip("Delta"))
    assert [b.name for b in radar.blips] == ["Alpha", "Delta"]


def test_radar_counts_grid_covers_every_quadrant_and_ring():
    radar = Radar(
        blips=[
            _blip("a", ring=Ring.ADOPT, quadrant=Quadrant.MODELS),
            _blip("b", ring=Ring.ADOPT, quadrant=Quadrant.MODELS),
            _blip("c", ring=Ring.HOLD, quadrant=Quadrant.INFRA),
        ]
    )
    grid = radar.counts()
    assert grid[Quadrant.MODELS][Ring.ADOPT] == 2
    assert grid[Quadrant.INFRA][Ring.HOLD] == 1
    # Every quadrant × ring cell exists, defaulting to zero.
    assert grid[Quadrant.TECHNIQUES][Ring.TRIAL] == 0
