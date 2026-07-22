"""The synthesis brain: JSON judge parsing, applying placements through the movement discipline
(create / move / refuse), feed pruning, deep-dive gating, and the digest — all with a fake LLM."""

from __future__ import annotations

import json
from datetime import date

from forge.radar.candidates import CandidateFeed, FeedEntry
from forge.radar.models import Blip, Quadrant, Radar, Ring
from forge.radar.sources.base import RawItem
from forge.radar.synthesis import (
    Placement,
    apply_placements,
    build_judge_messages,
    judge_candidates,
    parse_placements,
    render_digest,
    synthesize,
)
from forge.shared.llm import LLMConfig

CFG = LLMConfig(backend="openai")
D = date(2026, 7, 22)


def _entry(source: str, ext: str, title: str, **kw) -> FeedEntry:
    return FeedEntry.from_item(
        RawItem(source=source, external_id=ext, title=title, url=f"http://x/{ext}", **kw),
        today="2026-07-15",
    )


def _fake_complete(payload: dict):
    """A complete_fn that ignores the prompt and returns fixed JSON."""

    def _fn(cfg, *, system, user_message, model, max_tokens=8192):
        return json.dumps(payload)

    return _fn


# --- parsing -----------------------------------------------------------------


def test_parse_keeps_only_known_keys_and_coerces_enums():
    entries = [_entry("hackernews", "1", "Show HN: Foo agent framework")]
    raw = {
        "placements": [
            {
                "key": "hackernews:1",
                "keep": True,
                "name": "Foo",
                "quadrant": "agents & frameworks",
                "ring": "assess",
                "rationale": "worth watching",
            },
            {"key": "unknown:99", "keep": True, "name": "Ghost"},  # not a fed key → ignored
        ]
    }
    placements = parse_placements(raw, entries)
    assert len(placements) == 1
    p = placements[0]
    assert p.name == "Foo" and p.quadrant is Quadrant.AGENTS and p.ring is Ring.ASSESS


def test_parse_drops_hallucinated_enum_to_none():
    entries = [_entry("hackernews", "1", "t")]
    raw = {
        "placements": [
            {
                "key": "hackernews:1",
                "keep": True,
                "name": "X",
                "quadrant": "Nonsense",
                "ring": "Trial",
            }
        ]
    }
    p = parse_placements(raw, entries)[0]
    assert p.quadrant is None and p.ring is Ring.TRIAL
    assert not p.is_actionable()  # missing quadrant → not actionable


def test_judge_messages_include_current_blips_and_candidates():
    radar = Radar(
        blips=[
            Blip(
                name="vLLM",
                quadrant=Quadrant.INFRA,
                ring=Ring.ADOPT,
                first_seen="2026-01-01",
                last_seen="2026-01-01",
            )
        ]
    )
    entries = [_entry("arxiv", "2501.1", "A reasoning method")]
    system, user = build_judge_messages(entries, radar)
    assert "euclid" in system  # stack-personal
    assert "vLLM" in user and "arxiv:2501.1" in user


def test_judge_tolerates_trailing_model_chatter():
    # Some router models append prose after the JSON; the balanced-brace extractor must recover it.
    entries = [_entry("github", "a/b", "a/b")]
    noisy = (
        'Sure, here you go:\n{"placements": [{"key": "github:a/b", "keep": true, "name": "B", '
        '"quadrant": "Infra/Tooling", "ring": "Assess", "rationale": "r"}]}\n'
        "By the way, do you also want a README? {oops a stray brace}"
    )

    def fn(cfg, *, system, user_message, model, max_tokens=8192):
        return noisy

    placements = judge_candidates(entries, Radar(), complete_fn=fn, cfg=CFG, model="glm")
    assert len(placements) == 1 and placements[0].name == "B"


def test_judge_candidates_roundtrips_through_fake_llm():
    entries = [_entry("huggingface", "org/m", "org/m", quadrant_hint=Quadrant.MODELS)]
    fn = _fake_complete(
        {
            "placements": [
                {
                    "key": "huggingface:org/m",
                    "keep": True,
                    "name": "M",
                    "quadrant": "Models",
                    "ring": "Assess",
                    "rationale": "new open-weights model",
                },
            ]
        }
    )
    placements = judge_candidates(entries, Radar(), complete_fn=fn, cfg=CFG, model="research")
    assert placements[0].name == "M" and placements[0].ring is Ring.ASSESS


# --- apply -------------------------------------------------------------------


def test_apply_creates_a_new_blip_at_the_judged_ring():
    radar = Radar()
    entries = [_entry("hackernews", "1", "Show HN: Foo")]
    placements = [
        Placement(
            key="hackernews:1",
            keep=True,
            name="Foo",
            quadrant=Quadrant.AGENTS,
            ring=Ring.ASSESS,
            rationale="worth watching",
        )
    ]
    changes = apply_placements(radar, placements, entries, today=D)
    assert changes[0].kind == "create"
    blip = radar.get("Foo")
    assert blip.ring is Ring.ASSESS and blip.quadrant is Quadrant.AGENTS
    assert blip.links == ["http://x/1"]  # carried from the feed entry
    assert blip.evidence[0].source == "synthesis"


def test_apply_moves_an_existing_blip_through_the_discipline():
    radar = Radar(
        blips=[
            Blip(
                name="Foo",
                quadrant=Quadrant.AGENTS,
                ring=Ring.ASSESS,
                first_seen="2026-06-01",
                last_seen="2026-06-01",
            )
        ]
    )
    placements = [
        Placement(
            key="k",
            keep=True,
            name="Foo",
            quadrant=Quadrant.AGENTS,
            ring=Ring.TRIAL,
            rationale="trialed, works",
        )
    ]
    changes = apply_placements(radar, placements, [], today=D)
    assert changes[0].kind == "promote"
    assert radar.get("Foo").ring is Ring.TRIAL
    assert radar.get("Foo").ring_last is Ring.ASSESS


def test_apply_records_a_refused_move_without_changing_the_ring():
    # Blip moved yesterday → cooldown refuses today's move.
    radar = Radar(
        blips=[
            Blip(
                name="Foo",
                quadrant=Quadrant.AGENTS,
                ring=Ring.TRIAL,
                ring_last=Ring.ASSESS,
                first_seen="2026-06-01",
                last_seen="2026-07-21",
                last_moved="2026-07-21",
            )
        ]
    )
    placements = [
        Placement(
            key="k",
            keep=True,
            name="Foo",
            quadrant=Quadrant.AGENTS,
            ring=Ring.ADOPT,
            rationale="promote again",
        )
    ]
    changes = apply_placements(radar, placements, [], today=D)
    assert changes[0].kind == "refused"
    assert "cooldown" in changes[0].note
    assert radar.get("Foo").ring is Ring.TRIAL  # unchanged


def test_apply_same_ring_is_an_in_place_update():
    radar = Radar(
        blips=[
            Blip(
                name="Foo",
                quadrant=Quadrant.AGENTS,
                ring=Ring.ASSESS,
                first_seen="2026-06-01",
                last_seen="2026-06-01",
                rationale="old",
            )
        ]
    )
    placements = [
        Placement(
            key="k",
            keep=True,
            name="Foo",
            quadrant=Quadrant.AGENTS,
            ring=Ring.ASSESS,
            rationale="refreshed reason",
        )
    ]
    changes = apply_placements(radar, placements, [], today=D)
    assert changes[0].kind == "update"
    assert radar.get("Foo").rationale == "refreshed reason"


def test_apply_drop_is_recorded_and_creates_nothing():
    radar = Radar()
    placements = [Placement(key="k", keep=False, name="Junk", rationale="generic AI news")]
    changes = apply_placements(radar, placements, [], today=D)
    assert changes[0].kind == "drop"
    assert radar.blips == []


# --- orchestration -----------------------------------------------------------


def test_synthesize_applies_and_prunes_the_feed():
    radar = Radar()
    feed = CandidateFeed(
        entries=[
            _entry("hackernews", "1", "Show HN: Foo agent framework"),
            _entry("hackernews", "2", "A funding round"),
        ]
    )
    fn = _fake_complete(
        {
            "placements": [
                {
                    "key": "hackernews:1",
                    "keep": True,
                    "name": "Foo",
                    "quadrant": "Agents & Frameworks",
                    "ring": "Assess",
                    "rationale": "worth watching",
                },
                {"key": "hackernews:2", "keep": False, "name": "", "rationale": "not AI"},
            ]
        }
    )
    result, digest = synthesize(radar, feed, today=D, complete_fn=fn, cfg=CFG, model="research")

    assert result.judged == 2 and result.kept == 1
    assert radar.get("Foo") is not None
    assert feed.entries == []  # both judged candidates consumed
    assert "Foo" in digest and "New blips" in digest


def test_synthesize_deep_dive_only_runs_for_trial_or_adopt():
    radar = Radar()
    feed = CandidateFeed(
        entries=[
            _entry("hackernews", "1", "Assess thing"),
            _entry("hackernews", "2", "Trial thing"),
        ]
    )
    fn = _fake_complete(
        {
            "placements": [
                {
                    "key": "hackernews:1",
                    "keep": True,
                    "name": "AssessThing",
                    "quadrant": "Techniques",
                    "ring": "Assess",
                    "rationale": "watch",
                },
                {
                    "key": "hackernews:2",
                    "keep": True,
                    "name": "TrialThing",
                    "quadrant": "Techniques",
                    "ring": "Trial",
                    "rationale": "try it",
                },
            ]
        }
    )
    dived: list[str] = []

    def fake_dive(name, rationale):
        dived.append(name)
        return f"evidence about {name}"

    result, _ = synthesize(
        radar,
        feed,
        today=D,
        complete_fn=fn,
        cfg=CFG,
        model="research",
        deep=True,
        deep_dive_fn=fake_dive,
    )
    assert dived == ["TrialThing"]  # Assess not deep-dived
    assert "deep-dive" in radar.get("TrialThing").rationale
    assert result.deep_dived == ["TrialThing"]


def test_render_digest_groups_by_kind():
    from forge.radar.synthesis import Change, SynthesisResult

    result = SynthesisResult(
        judged=3,
        kept=2,
        changes=[
            Change(
                kind="create",
                name="Foo",
                quadrant=Quadrant.AGENTS,
                to_ring=Ring.ASSESS,
                rationale="new",
            ),
            Change(
                kind="promote",
                name="Bar",
                from_ring=Ring.ASSESS,
                to_ring=Ring.TRIAL,
                rationale="works",
            ),
        ],
    )
    digest = render_digest(result, today="2026-07-22")
    assert "## New blips" in digest and "## Promoted" in digest
    assert "Assess → Trial" in digest
