from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class DependabotSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DEPENDABOT_")

    branch_prefix: str = "deps"
    auto_log_path: Path = Path(__file__).parent / "logs" / "auto.jsonl"

    signoff_max_tokens: int = 4096
    signoff_timeout: float = 180.0
    scan_timeout: int = 120
    audit_timeout: int = 300
    metadata_timeout: float = 20.0
    max_candidates: int = 20


settings = DependabotSettings()
