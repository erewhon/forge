from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class EvalsSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EVALS_")

    goldsets_dir: Path = Path("/agents/evals/goldsets")
    runs_dir: Path = Path("/eval-runs")
    openai_base_url: str = "http://localhost:4010/v1"
    openai_api_key: str = "sk-local-router"
    model: str = "coder"
    repeats: int = 3
    temperature: float = 0.0
    timeout: float = 240.0
    max_tokens: int = 16_000


settings = EvalsSettings()
