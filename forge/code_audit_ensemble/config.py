from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class CodeAuditSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CODE_AUDIT_", env_file=".env", extra="ignore")

    openai_base_url: str = "http://localhost:4000/v1"
    openai_api_key: str = ""

    # Failover pools (strongest-first): finders + the dedup consolidator run through the router.
    finder_models: list[str] = ["coder", "qwen3.6-plus", "glm-5.1"]
    dedup_models: list[str] = ["coder", "qwen3.6-plus", "glm-5.1"]
    # The perspective-diverse skeptic panel; round-robined across the verify lenses.
    verify_models: list[str] = ["coder", "qwen3.6-plus"]
    verify_floor: int = 1  # min skeptics that must answer per finding, else the verdict is degraded

    concurrency: int = 3  # items/finders in flight; each verify item fans out to len(lenses) calls
    max_code_chars: int = 24000  # cap the code context fed to every finder
    max_tokens: int = 4096
    per_call_timeout: float = 150.0


settings = CodeAuditSettings()
