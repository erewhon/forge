from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class TestingEnsembleSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TESTING_ENSEMBLE_")

    openai_base_url: str = "http://localhost:4010/v1"
    openai_api_key: str = "sk-local-router"

    finder_models: list[str] = ["coder", "qwen3.6-plus", "glm-5.1"]
    dedup_models: list[str] = ["coder", "qwen3.6-plus", "glm-5.1"]
    verify_models: list[str] = ["coder", "qwen3.6-plus"]
    verify_floor: int = 1

    concurrency: int = 3
    max_context_chars: int = 28000  # split across the source + existing-tests sections
    source_fraction: float = 0.6  # how much of the budget the source gets; tests get the rest
    max_tokens: int = 4096
    per_call_timeout: float = 150.0


settings = TestingEnsembleSettings()
