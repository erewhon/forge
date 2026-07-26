from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from forge.shared.envfile import ENV_FILES
from forge.shared.llm import LLMConfig


class BookResearcherSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BOOK_RESEARCHER_", env_file=ENV_FILES, extra="ignore"
    )

    # Project paths
    project_dir: Path = Path.home() / "projects" / "book-research"

    # AI backend: "anthropic" or "openai" (for local/router endpoints)
    llm_backend: Literal["openai", "anthropic"] = "openai"

    # OpenAI-compatible models (used when llm_backend == "openai")
    openai_base_url: str = "http://localhost:4000/v1"
    openai_api_key: str = ""
    research_model: str = "research"  # router alias -> self-hosted model (local, private)
    synthesis_model: str = "coder"  # planning, verification, synthesis (also self-hosted)

    # Anthropic models (fallback when llm_backend == "anthropic")
    anthropic_model: str = "claude-sonnet-4-6"

    # Sprint settings
    max_sprints_per_run: int = 3
    score_threshold: int = 7  # minimum score (1-10) to accept findings
    max_findings_tokens: int = 4000  # truncate findings context for verifier

    # Adversarial verification panel (ensemble harness consumer #3): instead of one verifier, fan
    # out these diverse router models — each scores + challenges adversarially; scores are median-
    # aggregated (robust to a lenient/harsh outlier) and the challenges drive the next sprint. The
    # panel always runs through the router, even when llm_backend="anthropic".
    #
    # TEMPORARY (2026-07-26): partially de-localised, pending a GPU. The all-self-hosted panel
    # ["coder","gptoss","m2.7-local"] could not actually run — every non-Qwen self-hosted model
    # lives on hekaton, which is CPU-only (4x Xeon E7-4850 v2, no AVX2). Measured on a real sprint,
    # gptoss took 521s and m2.7-local >1600s against the panel's 120s timeout, so all three CPU
    # seats timed out on EVERY verification and only `coder` (graded twice) survived — nominally
    # 3-family, actually 1-family, with three lenses ungraded and no complaint in the output.
    #
    # So: keep the local Qwen seat and buy real family diversity from two vetted OpenCode Zen models
    # — glm=GLM-5.2 (Zhipu), m3=MiniMax-M3. Per Zen's docs those are zero-retention and not trained
    # on; the free tier (data may train the model) and the OpenAI/Anthropic routes (30-day
    # retention) are excluded, and test_config_privacy enforces both exclusions. Book findings DO
    # leave the homelab for verification under this config — an explicit, temporary trade for a
    # panel that functions. Revert to self-hosted-only once non-Qwen models can run on GPU.
    verifier_panel_models: list[str] = ["coder", "glm", "m3"]
    verifier_panel_floor: int = 2  # min members that must respond+parse, else degrade

    @property
    def sprints_dir(self) -> Path:
        return self.project_dir / "sprints"

    @property
    def knowledge_dir(self) -> Path:
        return self.project_dir / "knowledge"

    @property
    def outline_file(self) -> Path:
        return self.project_dir / "outline.yaml"

    def llm_cfg(self) -> LLMConfig:
        return LLMConfig(
            backend=self.llm_backend,
            openai_base_url=self.openai_base_url,
            openai_api_key=self.openai_api_key,
            anthropic_model=self.anthropic_model,
        )


settings = BookResearcherSettings()
