from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from forge.shared.envfile import ENV_FILES
from forge.shared.llm import LLMConfig


class GeneralResearcherSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GENERAL_RESEARCHER_", env_file=ENV_FILES, extra="ignore"
    )

    project_dir: Path = Path.home() / "projects" / "research"

    llm_backend: Literal["openai", "anthropic"] = "openai"
    openai_base_url: str = "http://localhost:4000/v1"
    openai_api_key: str = ""
    research_model: str = "research"  # router alias -> self-hosted model (local, private)
    synthesis_model: str = "coder"  # planning/synthesis (also self-hosted)
    anthropic_model: str = "claude-sonnet-4-6"

    max_sprints_per_run: int = 5
    score_threshold: int = 7
    max_findings_tokens: int = 4000

    # Adversarial verification panel: instead of one verifier, fan out these diverse router models
    # (harness consumer #3) — each scores + challenges adversarially, then scores are median-
    # aggregated (robust to a lenient/harsh outlier) and the challenges drive the next sprint. The
    # panel always runs through the router, even when llm_backend="anthropic". Never the
    # research_model itself (no self-grading).
    #
    # TEMPORARY (2026-07-26): partially de-localised, pending a GPU. The all-self-hosted panel
    # ["coder","gptoss","m2.7-local"] could not actually run — every non-Qwen self-hosted model
    # lives on hekaton, which is CPU-only (4x Xeon E7-4850 v2, no AVX2). Measured on a real sprint,
    # gptoss took 521s and m2.7-local >1600s against the panel's 120s timeout, so all three CPU
    # seats timed out on EVERY verification. What survived was `coder` graded twice — and the median
    # of two values is their mean, so "robust to a lenient/harsh outlier" was one model averaged
    # with itself, while the claim-verification / counter-narrative / actionability lenses went
    # ungraded entirely. Nominally 3-family, actually 1-family, and silent about it.
    #
    # So: keep the local Qwen seat and buy real family diversity from two vetted OpenCode Zen models
    # — glm=GLM-5.2 (Zhipu), m3=MiniMax-M3. Per Zen's docs those are zero-retention and not trained
    # on; the free tier (data may train the model) and the OpenAI/Anthropic routes (30-day
    # retention) are excluded, and test_config_privacy enforces both exclusions. Findings DO leave
    # the homelab for verification under this config — an explicit, temporary trade for a panel that
    # functions at all. Revert to self-hosted-only once non-Qwen models can run on GPU.
    verifier_panel_models: list[str] = ["coder", "glm", "m3"]
    verifier_panel_floor: int = 2  # min members that must respond+parse, else degrade

    # Synthesizer ensemble (research panel followup #2): instead of one model writing the final
    # answer, generate a candidate synthesis from each of these models, judge-pick the most
    # coherent, then graft in the unique key_sources / open_questions the runners-up surfaced. Runs
    # through the router. Floor 1 means a single parseable candidate is enough; 0 candidates falls
    # back to a single-model synthesis so the run always produces an answer. Both members are
    # self-hosted (models.yaml backend=vllm, never external) — synthesis stays local *deliberately*,
    # even while the verifier panel is temporarily on Zen: these are fast GPU Qwen aliases
    # (coder=Qwen3.6-35B hypatia, research=Qwen3-Next-80B archimedes) with no latency problem to
    # solve, so there is nothing to buy by sending findings off-box here. Keeping them local bounds
    # the exposure to the verification step alone. The ≥2-family diversity requirement applies to
    # the adversarial verifier, not this generator, so a single-family pair is fine here.
    synthesizer_panel_models: list[str] = ["coder", "research"]
    synthesizer_panel_floor: int = 1

    always_deepen: bool = False

    def llm_cfg(self) -> LLMConfig:
        return LLMConfig(
            backend=self.llm_backend,
            openai_base_url=self.openai_base_url,
            openai_api_key=self.openai_api_key,
            anthropic_model=self.anthropic_model,
        )


settings = GeneralResearcherSettings()
