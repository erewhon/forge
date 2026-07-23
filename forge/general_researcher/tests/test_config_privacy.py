"""Privacy guard: research panels must stay self-hosted (no cloud alias leaks findings).

Backs the "kept local for privacy" claim end-to-end — see the Forge task "Swap research and
book verifier panels to self-hosted models for privacy". Every configured panel alias must be a
known self-hosted router alias (models.yaml backend=vllm/lmstudio, never external), and the
adversarial verifier must keep >=2 distinct model families so it stays a real cross-check.
"""

from __future__ import annotations

from forge.general_researcher.config import GeneralResearcherSettings

# Router aliases whose models.yaml backend is self-hosted (vllm/lmstudio). Family tags drive the
# adversarial-diversity check. Keep in sync with ~/Projects/erewhon/llm-router/models.yaml.
SELF_HOSTED_FAMILY: dict[str, str] = {
    # Qwen (GPU nodes)
    "coder": "qwen",
    "research": "qwen",
    "thinker": "qwen",
    "qwen3.6-hypatia": "qwen",
    "qwen3.6-local": "qwen",
    "coder-next": "qwen",
    "coder-next-local": "qwen",
    # gpt-oss (hekaton CPU)
    "gptoss": "gpt-oss",
    "gpt-oss": "gpt-oss",
    "gpt-oss-120b-local": "gpt-oss",
    # MiniMax (hekaton CPU)
    "m2.7-local": "minimax",
    "minimax-local": "minimax",
    # Ling (hekaton CPU)
    "ling-flash": "ling",
    "ling": "ling",
    "ling-flash-local": "ling",
}

# Known cloud (external-backend) aliases — findings would leave the homelab. Belt-and-suspenders
# alongside the allowlist check. Note the *-local MiniMax aliases are self-hosted, unlike these.
KNOWN_CLOUD_ALIASES: set[str] = {
    "glm", "glm-5.1", "glm-5.2",
    "qwen3.6-plus", "qwen-plus", "qwen3.6-cloud", "qwen3.7-plus", "qwen3.7-cloud",
    "kimi", "kimi-code", "k2.6", "k2.7", "k3",
    "kimi-k2.6", "kimi-k2.7", "kimi-k2.7-code", "kimi-k3",
    "m3", "minimax-m3", "minimax-m2.7", "m2.7",
    "sonnet", "opus", "fable",
    "claude-sonnet-4-6", "claude-opus-4-8", "claude-sonnet-5",
    "deepseek", "nemotron-ultra", "nemotron-3-ultra-free",
}

settings = GeneralResearcherSettings()


def test_verifier_panel_is_self_hosted_only() -> None:
    for alias in settings.verifier_panel_models:
        assert alias in SELF_HOSTED_FAMILY, f"{alias!r} is not a known self-hosted alias"
        assert alias not in KNOWN_CLOUD_ALIASES, f"{alias!r} is a cloud alias — findings would leak"


def test_synthesizer_panel_is_self_hosted_only() -> None:
    for alias in settings.synthesizer_panel_models:
        assert alias in SELF_HOSTED_FAMILY, f"{alias!r} is not a known self-hosted alias"
        assert alias not in KNOWN_CLOUD_ALIASES, f"{alias!r} is a cloud alias — findings would leak"


def test_verifier_panel_keeps_family_diversity() -> None:
    families = {SELF_HOSTED_FAMILY[a] for a in settings.verifier_panel_models}
    assert len(families) >= 2, f"adversarial panel needs >=2 distinct families, got {families}"


def test_verifier_panel_excludes_research_model_no_self_grading() -> None:
    # research_model resolves to the same self-hosted model as the "research"/"thinker" aliases;
    # keeping any of them out of the verifier panel prevents the research model grading itself.
    banned = {settings.research_model, "research", "thinker"}
    overlap = set(settings.verifier_panel_models) & banned
    assert not overlap, f"no self-grading: research model {overlap} in verifier panel"


def test_panels_satisfy_their_floors() -> None:
    assert len(settings.verifier_panel_models) >= settings.verifier_panel_floor
    assert len(settings.synthesizer_panel_models) >= settings.synthesizer_panel_floor
