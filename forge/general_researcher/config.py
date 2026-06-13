from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from agents.shared.llm import LLMConfig


class GeneralResearcherSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GENERAL_RESEARCHER_")

    project_dir: Path = Path.home() / "Projects" / "erewhon" / "meta" / "research"

    llm_backend: Literal["openai", "anthropic"] = "openai"
    openai_base_url: str = "http://localhost:4010/v1"
    openai_api_key: str = "sk-local-router"
    research_model: str = "research"
    synthesis_model: str = "coder"
    anthropic_model: str = "claude-sonnet-4-6"

    max_sprints_per_run: int = 5
    score_threshold: int = 7
    max_findings_tokens: int = 4000

    always_deepen: bool = False

    def llm_cfg(self) -> LLMConfig:
        return LLMConfig(
            backend=self.llm_backend,
            openai_base_url=self.openai_base_url,
            openai_api_key=self.openai_api_key,
            anthropic_model=self.anthropic_model,
        )


settings = GeneralResearcherSettings()
