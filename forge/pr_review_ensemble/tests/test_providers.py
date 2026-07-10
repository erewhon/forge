"""The anthropic reviewer seat: routed through the local LiteLLM proxy by default, native SDK when
``anthropic_base_url`` is cleared. Routing through the router removes the per-shell
ANTHROPIC_API_KEY dependency that left the epic-gate seat mute (quorum 0/2) while keeping the seat's
Claude-family identity for quorum/diversity accounting.
"""

from __future__ import annotations

from forge.pr_review_ensemble import providers
from forge.pr_review_ensemble.config import settings


def test_anthropic_slot_routes_through_router_by_default(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_enabled", True)
    monkeypatch.setattr(settings, "anthropic_base_url", "http://llm-router.internal:4000/v1")
    monkeypatch.setattr(settings, "anthropic_api_key", "test-router-key")
    monkeypatch.setattr(settings, "anthropic_model", "claude-sonnet-5")

    slot = providers._anthropic_slot()

    assert slot.active
    assert slot.provider == "anthropic"  # still the Claude-family seat, just proxied
    executor = slot.pool.executors[0]
    assert executor.kind == "openai"  # OpenAI-compat path, not the native SDK
    assert executor.base_url == "http://llm-router.internal:4000/v1"
    assert executor.api_key == "test-router-key"
    assert executor.model == "claude-sonnet-5"


def test_anthropic_slot_falls_back_to_native_sdk_when_base_url_empty(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_enabled", True)
    monkeypatch.setattr(settings, "anthropic_base_url", "")

    slot = providers._anthropic_slot()

    assert slot.active
    executor = slot.pool.executors[0]
    assert executor.kind == "anthropic"  # native SDK (reads ANTHROPIC_API_KEY from env)
    assert executor.base_url is None


def test_anthropic_slot_skipped_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "anthropic_enabled", False)

    slot = providers._anthropic_slot()

    assert not slot.active
    assert slot.skipped_reason == "disabled in config"
