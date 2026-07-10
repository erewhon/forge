from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class RefactorEnsembleSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="REFACTOR_ENSEMBLE_", env_file=".env", extra="ignore"
    )

    openai_base_url: str = "http://localhost:4000/v1"
    openai_api_key: str = ""

    finder_models: list[str] = ["coder", "qwen3.6-plus", "glm-5.1"]
    dedup_models: list[str] = ["coder", "qwen3.6-plus", "glm-5.1"]
    verify_models: list[str] = ["coder", "qwen3.6-plus"]
    verify_floor: int = 1

    concurrency: int = 3
    max_code_chars: int = 24000  # cap the code context fed to every finder
    max_tokens: int = 4096
    per_call_timeout: float = 150.0


settings = RefactorEnsembleSettings()
