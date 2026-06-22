from __future__ import annotations

from pathlib import Path

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

    # Auto-merge loop (`meta testing --auto`): generate tests for confirmed gaps, gate, then act.
    gen_models: list[str] = ["coder", "qwen3.6-plus"]  # failover pool for writing the tests
    gen_max_tokens: int = 8192  # test code can be long; give it room
    auto_max_gaps: int = 3  # cap generated tests per run (blast-radius control)
    signoff_max_tokens: int = 1024  # each gatekeeper returns a small JSON verdict
    signoff_timeout: float = 120.0
    auto_log_path: Path = Path(__file__).parent / "logs" / "auto.jsonl"


settings = TestingEnsembleSettings()
