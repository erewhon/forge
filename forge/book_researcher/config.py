from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from agents.shared.llm import LLMConfig


class BookResearcherSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BOOK_RESEARCHER_")

    # Project paths
    project_dir: Path = Path.home() / "Projects" / "erewhon" / "meta" / "book-research"

    # AI backend: "anthropic" or "openai" (for local/router endpoints)
    llm_backend: Literal["openai", "anthropic"] = "openai"

    # OpenAI-compatible models (used when llm_backend == "openai")
    openai_base_url: str = "http://localhost:4010/v1"
    openai_api_key: str = "sk-local-router"
    research_model: str = "research"  # 27B model, VPN-routed for privacy
    synthesis_model: str = "coder"  # planning, verification, synthesis

    # Anthropic models (fallback when llm_backend == "anthropic")
    anthropic_model: str = "claude-sonnet-4-6"

    # Sprint settings
    max_sprints_per_run: int = 3
    score_threshold: int = 7  # minimum score (1-10) to accept findings
    max_findings_tokens: int = 4000  # truncate findings context for verifier

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
