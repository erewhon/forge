"""Privacy guard: the book verifier panel may only use vetted models.

Mirror of the general researcher's guard — see that file for the full rationale. In short: the
"self-hosted only" rule became unrunnable (every non-Qwen self-hosted model is on hekaton, CPU-only,
where a seat needs 500-1600s against a 120s timeout), so pending a GPU the panel may include vetted
OpenCode Zen models alongside a local seat.

Vetted per https://opencode.ai/docs/zen/ means US-hosted, zero-retention, not trained on. The two
carve-outs in that policy are hard-banned: the **free/stealth tier** (data "may be used to improve
the model"; docs warn against submitting confidential data) and the **OpenAI/Anthropic routes**
(requests "retained for 30 days").
"""

from __future__ import annotations

from forge.book_researcher.config import BookResearcherSettings

# Self-hosted router aliases (models.yaml backend=vllm/lmstudio), tagged by family.
SELF_HOSTED_FAMILY: dict[str, str] = {
    "coder": "qwen",
    "qwen3.6-hypatia": "qwen",
    "qwen3.6-local": "qwen",
    "coder-next": "qwen",
    "coder-next-local": "qwen",
    "gptoss": "gpt-oss",
    "gpt-oss": "gpt-oss",  # delphi GPU since 2026-08-18 (was hekaton CPU)
    "gpt-oss-120b-local": "gpt-oss",
    # Role aliases track the router's CURRENT resolution and go stale when models.yaml flips a
    # role — that staleness let a Qwen seat family-self-grade for five days in Aug 2026. Until the
    # map is derived from models.yaml (filed task), update these WITH every role flip:
    # research → qwen3-coder-next (Qwen) since 2026-08-13; thinker → Lightning (NVIDIA) 2026-08-18.
    "research": "qwen",
    "thinker": "nvidia",
    # Ling 3 flash (hekaton CPU, seated 2026-08-22) and Gemma 4 26B (talos B70 card 1,
    # seated 2026-08-22) — the 2026-08-22 review-roster rewire seats
    "ling": "bailing",
    "ling-3-flash-slow": "bailing",
    "gemma": "google",
    "gemma4-26b": "google",
    # Nemotron (talos B70 GPU)
    "lightning": "nvidia",
    "nemotron-3.5-lightning": "nvidia",
    "minimax": "minimax",
    "minimax-m2.7-reap": "minimax",
    "m2.7-local": "minimax",
    "minimax-local": "minimax",
    # Ling flash 2.0 aliases share Ling 3's family (both bailingmoe).
    "ling-flash": "bailing",
    "ling-flash-local": "bailing",
    # Router aliases for the seats above, as the outline pool names them.
    "gemma4": "google",
    "gemma-4": "google",
    "nemotron-lightning": "nvidia",
}

# Vetted OpenCode Zen aliases: paid, zero-retention, not trained on, not an OpenAI/Anthropic route.
VETTED_ZEN_FAMILY: dict[str, str] = {
    "glm": "zhipu",  # glm-5.2
    "m3": "minimax",  # minimax-m3
    "kimi": "moonshot",  # kimi-k2.7-code
    "k2.7": "moonshot",
    "kimi-k2.7": "moonshot",
    "kimi-code": "moonshot",
}

ALLOWED_FAMILY: dict[str, str] = {**SELF_HOSTED_FAMILY, **VETTED_ZEN_FAMILY}

# Free / stealth tier — data may be used to improve the model.
FREE_TIER_ALIASES: set[str] = {
    "big-pickle",
    "nemotron-ultra",
    "nemotron-3-ultra-free",
    "north-mini-code-free",
    "deepseek-v4-flash-free",
    "mimo-v2.5-free",
    "laguna-s-2.1-free",
    "ling-3.0-flash-free",
    "auto-free",
}

# OpenAI / Anthropic routes — requests retained 30 days.
RETAINING_ROUTE_ALIASES: set[str] = {
    "sonnet",
    "opus",
    "fable",
    "claude-sonnet-4-6",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-haiku-4-5",
    "claude-fable-5",
    "anthropic-gateway",
    "gpt-5",
    "gpt-5-mini",
    "openai-gateway",
}

settings = BookResearcherSettings()


def test_verifier_panel_uses_only_vetted_models() -> None:
    for alias in settings.verifier_panel_models:
        assert alias in ALLOWED_FAMILY, f"{alias!r} is not a vetted alias"
        assert alias not in FREE_TIER_ALIASES, f"{alias!r} is free-tier — may train on data"
        assert alias not in RETAINING_ROUTE_ALIASES, f"{alias!r} retains requests 30 days"


def test_verifier_panel_keeps_family_diversity() -> None:
    families = {ALLOWED_FAMILY[a] for a in settings.verifier_panel_models}
    assert len(families) >= 2, f"adversarial panel needs >=2 distinct families, got {families}"


def test_verifier_panel_keeps_a_self_hosted_seat() -> None:
    """At least one seat stays local, so the panel never becomes wholly off-box."""
    local = [a for a in settings.verifier_panel_models if a in SELF_HOSTED_FAMILY]
    assert local, "verifier panel has no self-hosted seat"


def test_verifier_panel_excludes_research_model_no_self_grading() -> None:
    banned = {settings.research_model, "research", "thinker"}
    overlap = set(settings.verifier_panel_models) & banned
    assert not overlap, f"no self-grading: research model {overlap} in verifier panel"


def test_verifier_panel_excludes_research_models_family() -> None:
    """Family-level no-self-grading — see the general researcher's twin test for the rationale."""
    research_family = ALLOWED_FAMILY.get(settings.research_model)
    assert research_family is not None, f"unknown research model {settings.research_model!r}"
    offenders = {a for a in settings.verifier_panel_models if ALLOWED_FAMILY[a] == research_family}
    assert not offenders, f"panel seats {offenders} share the research model's family"


def test_verifier_panel_satisfies_floor() -> None:
    assert len(settings.verifier_panel_models) >= settings.verifier_panel_floor


def test_local_lane_panel_is_fully_self_hosted() -> None:
    """The privacy lane's whole point: no seat may resolve off-box. Family purity is an accepted
    trade in this lane (see the general researcher's config) — self-hosting is not negotiable."""
    for alias in settings.verifier_panel_models_local:
        assert alias in SELF_HOSTED_FAMILY, f"local-lane seat {alias!r} is not self-hosted"


def test_active_panel_routes_by_lane() -> None:
    local = BookResearcherSettings(panel_lane="local")
    assert local.active_verifier_panel() == local.verifier_panel_models_local
    default = BookResearcherSettings()
    assert default.active_verifier_panel() == default.verifier_panel_models
    bogus = BookResearcherSettings(panel_lane="bogus")
    assert bogus.active_verifier_panel() == default.verifier_panel_models


def test_outline_pool_uses_only_vetted_models() -> None:
    """The outline pool (lint --critic / revise / decompose) reads the book's reviews and
    outline; same vetting as the panel."""
    for alias in settings.outline_models:
        assert alias in ALLOWED_FAMILY, f"{alias!r} is not a vetted alias"
        assert alias not in FREE_TIER_ALIASES, f"{alias!r} is free-tier — may train on data"
        assert alias not in RETAINING_ROUTE_ALIASES, f"{alias!r} retains requests 30 days"


def test_outline_pool_keeps_a_self_hosted_seat() -> None:
    assert any(a in SELF_HOSTED_FAMILY for a in settings.outline_models)


# --- X-Router-Privacy: the lane is enforced on the wire, not just by roster choice ---


def test_privacy_tier_follows_the_lane() -> None:
    """local lane → `local` (the router refuses any off-box candidate, overflow included); the
    default lane → `zdr` (vetted seats served from a zero-retention-enforceable endpoint; a
    chain's Zen member is skipped). Never `any` by default — that is the pre-header behaviour
    and has to be asked for."""
    assert BookResearcherSettings(panel_lane="local").privacy_tier() == "local"
    assert BookResearcherSettings().privacy_tier() == "zdr"
    assert BookResearcherSettings(panel_lane="bogus").privacy_tier() == "zdr"


def test_privacy_override_wins_over_the_lane() -> None:
    assert BookResearcherSettings(panel_lane="local", panel_privacy="any").privacy_tier() == "any"
    assert BookResearcherSettings(panel_privacy="local").privacy_tier() == "local"


def test_every_router_call_in_the_run_carries_the_tier() -> None:
    """The research model / planner path (LLMConfig) and the panel path share one tier."""
    local = BookResearcherSettings(panel_lane="local")
    assert local.llm_cfg().privacy == "local"
    assert BookResearcherSettings().llm_cfg().privacy == "zdr"


def test_local_lane_refuses_the_native_anthropic_backend() -> None:
    """A local-lane run on the native Anthropic backend would ship every sprint to Anthropic
    while claiming nothing left the homelab. Fail closed; 'any' states the trade explicitly."""
    import pytest

    with pytest.raises(ValueError, match="cannot honour"):
        BookResearcherSettings(panel_lane="local", llm_backend="anthropic").llm_cfg()
    assert (
        BookResearcherSettings(llm_backend="anthropic", panel_privacy="any").llm_cfg().privacy
        == "any"
    )
