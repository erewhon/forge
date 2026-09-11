"""The reviewer roster: three primary seats (sonnet-5, ling3, gemma), each a failover Pool
whose backup is pulled in only when the primary is down, and every model routed through the LLM
router so the whole roster is one endpoint + key. The sonnet seat keeps the Claude-family identity
(proxied by default, native SDK when ``anthropic_base_url`` is cleared) and its
``anthropic_enabled`` toggle. The all-local roster (ling3/gemma/lightning) must stay
fully router-local — no Anthropic executor anywhere in its chains.
"""

from __future__ import annotations

from forge.pr_review_ensemble import providers
from forge.pr_review_ensemble.config import settings


def _route_to_router(monkeypatch):
    monkeypatch.setattr(settings, "local_base_url", "http://router.internal:4010/v1")
    monkeypatch.setattr(settings, "local_api_key", "sk-router")


def test_build_reviewer_slots_is_the_three_seat_roster(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_enabled", True)
    slots = providers.build_reviewer_slots()
    assert [s.provider for s in slots] == ["sonnet-5", "ling3", "gemma"]
    assert all(s.active for s in slots)


def test_each_seat_is_a_failover_chain_with_the_expected_backup(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_enabled", True)
    monkeypatch.setattr(settings, "anthropic_base_url", "http://router.internal:4010/v1")
    _route_to_router(monkeypatch)

    sonnet, ling3, gemma = providers.build_reviewer_slots()

    # sonnet-5: Claude primary + local coder break-glass backup.
    assert [e.model for e in sonnet.pool.executors] == [settings.anthropic_model, "coder"]
    # ling3: slow hekaton primary + fast local Lightning, then kimi (cloud) — a hekaton outage
    # degrades to a fast local reviewer, not straight to a metered seat. gemma: B70 primary +
    # gpt-oss (the demoted local executor) then glm (cloud). All via the router.
    assert [e.model for e in ling3.pool.executors] == ["ling", "lightning", "kimi"]
    assert [e.model for e in gemma.pool.executors] == ["gemma", "gpt-oss", "glm"]


def test_local_seats_and_backups_route_through_the_router(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_enabled", True)
    monkeypatch.setattr(settings, "anthropic_base_url", "http://router.internal:4010/v1")
    _route_to_router(monkeypatch)

    _, ling3, gemma = providers.build_reviewer_slots()

    for slot in (ling3, gemma):
        for ex in slot.pool.executors:
            assert ex.kind == "openai"
            assert ex.base_url == "http://router.internal:4010/v1"
            assert ex.api_key == "sk-router"


def test_roster_for_lane_routes_local_and_fails_closed_to_frontier(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_enabled", True)
    _route_to_router(monkeypatch)

    local = providers.roster_for_lane("local")
    assert [s.provider for s in local] == ["ling3", "gemma", "lightning"]
    assert [s.provider for s in providers.roster_for_lane("frontier")][0] == "sonnet-5"
    # An unknown lane value gets the expensive-but-safer frontier roster, never local.
    assert [s.provider for s in providers.roster_for_lane("bogus")][0] == "sonnet-5"


def test_local_roster_is_the_three_family_trio_and_fully_local(monkeypatch):
    _route_to_router(monkeypatch)

    slots = providers.build_local_reviewer_slots()

    assert [s.provider for s in slots] == ["ling3", "gemma", "lightning"]
    assert all(s.active for s in slots)
    # Lightning keeps its Zen backups (m3/kimi) — cloud, but only reached on failover; the
    # primaries are all self-hosted. Fully local-first means every executor in every chain is
    # a router-backed OpenAI-compat one — no Anthropic executor can appear anywhere.
    lightning = slots[2]
    assert [e.model for e in lightning.pool.executors] == ["lightning", "m3", "kimi"]
    for slot in slots:
        for ex in slot.pool.executors:
            assert ex.kind == "openai"
            assert ex.base_url == "http://router.internal:4010/v1"


# --- X-Router-Privacy per roster ---


def test_local_roster_sends_local_on_every_seat_and_backup(monkeypatch):
    """The header is what makes 'every diff stays in the homelab' true on the wire: the cloud
    backups (m3/kimi/glm) are still listed, but under `local` the router refuses them (403,
    terminal) instead of quietly serving a routine-lane diff from the cloud."""
    _route_to_router(monkeypatch)
    for slot in providers.build_local_reviewer_slots():
        assert {ex.privacy for ex in slot.pool.executors} == {"local"}


def test_frontier_roster_sends_any_by_default(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_enabled", True)
    monkeypatch.setattr(settings, "anthropic_base_url", "http://router.internal:4010/v1")
    _route_to_router(monkeypatch)
    for slot in providers.build_reviewer_slots():
        assert {ex.privacy for ex in slot.pool.executors} == {"any"}


def test_roster_tier_is_configurable_per_lane(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_enabled", True)
    monkeypatch.setattr(settings, "anthropic_base_url", "http://router.internal:4010/v1")
    _route_to_router(monkeypatch)
    monkeypatch.setattr(settings, "frontier_privacy", "zdr")
    for slot in providers.roster_for_lane("frontier"):
        assert {ex.privacy for ex in slot.pool.executors} == {"zdr"}
    # An explicit argument wins over the setting.
    for slot in providers.build_reviewer_slots(privacy="local"):
        assert {ex.privacy for ex in slot.pool.executors} == {"local"}


def test_native_sdk_sonnet_seat_refuses_a_strict_tier(monkeypatch):
    """With anthropic_base_url cleared the sonnet primary is the native SDK, which bypasses the
    router — it cannot honour local/zdr, and the roster must fail loudly rather than build a
    seat that silently ships the diff to Anthropic's retaining API."""
    import pytest

    monkeypatch.setattr(settings, "anthropic_enabled", True)
    monkeypatch.setattr(settings, "anthropic_base_url", "")
    monkeypatch.setattr(settings, "frontier_privacy", "zdr")
    with pytest.raises(ValueError, match="bypasses the router"):
        providers.build_reviewer_slots()


def test_sonnet_primary_routes_through_router_by_default(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_enabled", True)
    monkeypatch.setattr(settings, "anthropic_base_url", "http://router.internal:4010/v1")
    monkeypatch.setattr(settings, "anthropic_api_key", "test-router-key")
    monkeypatch.setattr(settings, "anthropic_model", "claude-sonnet-5")

    primary = providers._sonnet_slot(privacy="any").pool.executors[0]

    assert primary.kind == "openai"  # OpenAI-compat proxy path, not the native SDK
    assert primary.base_url == "http://router.internal:4010/v1"
    assert primary.api_key == "test-router-key"
    assert primary.model == "claude-sonnet-5"


def test_sonnet_primary_falls_back_to_native_sdk_when_base_url_empty(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_enabled", True)
    monkeypatch.setattr(settings, "anthropic_base_url", "")

    primary = providers._sonnet_slot(privacy="any").pool.executors[0]

    assert primary.kind == "anthropic"  # native SDK (reads ANTHROPIC_API_KEY from env)
    assert primary.base_url is None


def test_sonnet_seat_skipped_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_enabled", False)

    slot = providers._sonnet_slot(privacy="any")

    assert not slot.active
    assert slot.skipped_reason == "disabled in config"
    # Still in the roster (attempted-but-skipped for quorum accounting); local seats stay active.
    labels = [(s.provider, s.active) for s in providers.build_reviewer_slots()]
    assert labels == [("sonnet-5", False), ("ling3", True), ("gemma", True)]
