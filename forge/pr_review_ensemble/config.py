from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class PRReviewEnsembleSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PR_REVIEW_ENSEMBLE_")

    # Anthropic provider
    anthropic_enabled: bool = True
    anthropic_model: str = "claude-sonnet-4-6"
    anthropic_max_tokens: int = 4096

    # Local LLM router (LiteLLM on Euclid)
    local_enabled: bool = True
    local_base_url: str = "http://localhost:4010/v1"
    local_api_key: str = "sk-local-router"
    local_model: str = "coder"
    local_max_tokens: int = 4096

    # OpenCode Zen (configure via env vars; gracefully skipped if api key empty)
    opencode_zen_enabled: bool = True
    opencode_zen_base_url: str = "https://opencode.ai/zen/v1"
    opencode_zen_api_key: str = ""
    opencode_zen_model: str = "kimi-k2.6"
    opencode_zen_max_tokens: int = 4096

    # Aggregator: preferred synthesizer, tried first. The aggregator runs through a failover
    # Pool whose rotation is [preferred, then anthropic -> opencode_zen -> local]; if every
    # member is down, AggregateCombiner falls back to deterministic concatenation. "local" is
    # the structural break-glass (always reachable), so it sits last in the rotation.
    aggregator_provider: str = "anthropic"
    aggregator_max_tokens: int = 4096

    # Runner
    per_provider_timeout_seconds: float = 120.0
    review_max_tokens: int = (
        4096  # max_tokens for each reviewer's pass (Prompt-level in the harness)
    )
    quorum_floor: int = 2

    # Logging
    log_path: Path = Path(__file__).parent / "logs" / "runs.jsonl"


settings = PRReviewEnsembleSettings()
