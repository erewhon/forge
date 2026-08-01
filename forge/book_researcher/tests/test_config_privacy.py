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
    "gpt-oss": "gpt-oss",
    "gpt-oss-120b-local": "gpt-oss",
    # thinker/research → MiniMax-M2.7-REAP on archimedes GPU since 2026-07-31 (was Qwen3-Next-80B).
    "research": "minimax",
    "thinker": "minimax",
    "minimax": "minimax",
    "minimax-m2.7-reap": "minimax",
    "m2.7-local": "minimax",
    "minimax-local": "minimax",
    "ling-flash": "ling",
    "ling": "ling",
    "ling-flash-local": "ling",
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
