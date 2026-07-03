from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from agents.shared.llm import LLMConfig


class CodingPipelineSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CODING_PIPELINE_")

    # Per-epic run dirs (framing, tree, wave records, journal) live here.
    runs_dir: Path = Path.home() / "Projects" / "erewhon" / "meta" / "pipeline-runs"

    # Loop bounds. max_waves is per *run* (re-run to continue, like the research harness's
    # --max-sprints); max_leaf_attempts is per leaf across the whole epic — at the cap the leaf
    # escalates to a human instead of retrying blind.
    max_waves: int = 3
    wave_size: int = 4
    max_leaf_attempts: int = 2

    # All epic work lands on {branch_prefix}/{epic_slug}; main only moves at the epic gate.
    branch_prefix: str = "pipeline"

    # Architect LLM (strong tier) — the headless path only. The interactive Fable session IS the
    # architect during the free window and never touches these.
    llm_backend: Literal["openai", "anthropic"] = "openai"
    openai_base_url: str = "http://localhost:4010/v1"
    openai_api_key: str = "sk-local-router"
    architect_model: str = "coder"
    anthropic_model: str = "claude-sonnet-4-6"

    def llm_cfg(self) -> LLMConfig:
        return LLMConfig(
            backend=self.llm_backend,
            openai_base_url=self.openai_base_url,
            openai_api_key=self.openai_api_key,
            anthropic_model=self.anthropic_model,
        )


settings = CodingPipelineSettings()
