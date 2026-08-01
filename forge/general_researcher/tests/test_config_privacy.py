"""Privacy guard: research panels may only use vetted models.

The rule used to be "self-hosted only". That became unrunnable — every non-Qwen self-hosted model
is on hekaton (CPU-only, no AVX2), where a panel seat needs 500-1600s against a 120s timeout, so the
nominally 3-family panel silently collapsed to one model grading twice. Pending a GPU, the verifier
panel may include **vetted OpenCode Zen** models alongside a local seat.

Vetted means, per Zen's published policy (https://opencode.ai/docs/zen/): US-hosted, zero-retention,
not trained on. The two carve-outs in that policy are hard-banned here:

* **free / stealth tier** — collected data "may be used to improve the model", and the docs say
  outright not to submit confidential data. Research findings are exactly that.
* **OpenAI / Anthropic routes** — requests "retained for 30 days". Not zero-retention.

The synthesizer panel is held to the stricter, original rule: fully self-hosted. It has no latency
problem to solve, so there is nothing to buy by sending findings off-box, and keeping it local
bounds the exposure to the verification step alone.
"""

from __future__ import annotations

from forge.general_researcher.config import GeneralResearcherSettings

# Self-hosted router aliases (models.yaml backend=vllm/lmstudio), tagged by family.
SELF_HOSTED_FAMILY: dict[str, str] = {
    # Qwen (GPU nodes)
    "coder": "qwen",
    "qwen3.6-hypatia": "qwen",
    "qwen3.6-local": "qwen",
    "coder-next": "qwen",
    "coder-next-local": "qwen",
    # gpt-oss (hekaton CPU)
    "gptoss": "gpt-oss",
    "gpt-oss": "gpt-oss",
    "gpt-oss-120b-local": "gpt-oss",
    # MiniMax — thinker/research moved from Qwen3-Next-80B to MiniMax-M2.7-REAP on
    # archimedes GPU 2026-07-31; m2.7-local is the full 256-expert build on hekaton CPU.
    "research": "minimax",
    "thinker": "minimax",
    "minimax": "minimax",
    "minimax-m2.7-reap": "minimax",
    "m2.7-local": "minimax",
    "minimax-local": "minimax",
    # Ling (hekaton CPU)
    "ling-flash": "ling",
    "ling": "ling",
    "ling-flash-local": "ling",
}

# Vetted OpenCode Zen aliases: paid, zero-retention, not trained on, and NOT an OpenAI/Anthropic
# route. Temporary allowance pending a GPU — see the config comment. Add to this only after
# checking the model against Zen's policy page.
VETTED_ZEN_FAMILY: dict[str, str] = {
    "glm": "zhipu",  # glm-5.2
    "m3": "minimax",  # minimax-m3
    "kimi": "moonshot",  # kimi-k2.7-code
    "k2.7": "moonshot",
    "kimi-k2.7": "moonshot",
    "kimi-code": "moonshot",
}

ALLOWED_FAMILY: dict[str, str] = {**SELF_HOSTED_FAMILY, **VETTED_ZEN_FAMILY}

# Free / stealth tier — Zen's docs say collected data may be used to improve the model, and warn
# against submitting confidential data. Never acceptable for research findings.
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

# OpenAI / Anthropic routes — requests retained 30 days. Not zero-retention.
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

settings = GeneralResearcherSettings()


def _assert_allowed(aliases: list[str], where: str) -> None:
    for alias in aliases:
        assert alias in ALLOWED_FAMILY, f"{where}: {alias!r} is not a vetted alias"
        assert alias not in FREE_TIER_ALIASES, (
            f"{where}: {alias!r} is free-tier — may train on data"
        )
        assert alias not in RETAINING_ROUTE_ALIASES, f"{where}: {alias!r} retains requests 30 days"


def test_verifier_panel_uses_only_vetted_models() -> None:
    _assert_allowed(settings.verifier_panel_models, "verifier panel")


def test_synthesizer_panel_stays_self_hosted() -> None:
    # Stricter than the verifier on purpose: synthesis has no latency problem, so it stays local.
    for alias in settings.synthesizer_panel_models:
        assert alias in SELF_HOSTED_FAMILY, f"synthesizer panel: {alias!r} is not self-hosted"


def test_no_free_tier_anywhere() -> None:
    every = set(settings.verifier_panel_models + settings.synthesizer_panel_models)
    assert not (every & FREE_TIER_ALIASES)


def test_no_thirty_day_retention_routes_anywhere() -> None:
    every = set(settings.verifier_panel_models + settings.synthesizer_panel_models)
    assert not (every & RETAINING_ROUTE_ALIASES)


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
    """The family-level form of no-self-grading. Exact-alias exclusion missed a soft variant:
    when thinker/research became MiniMax (2026-07-31), the then-panel-member m3=MiniMax-M3 had a
    MiniMax model grading MiniMax research output — family-correlated leniency the median can't
    fully wash out with only 3 seats. Ban the research model's whole family from the panel."""
    research_family = ALLOWED_FAMILY.get(settings.research_model)
    assert research_family is not None, f"unknown research model {settings.research_model!r}"
    offenders = {a for a in settings.verifier_panel_models if ALLOWED_FAMILY[a] == research_family}
    assert not offenders, f"panel seats {offenders} share the research model's family"


def test_panels_satisfy_their_floors() -> None:
    assert len(settings.verifier_panel_models) >= settings.verifier_panel_floor
    assert len(settings.synthesizer_panel_models) >= settings.synthesizer_panel_floor
