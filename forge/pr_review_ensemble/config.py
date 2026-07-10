from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class PRReviewEnsembleSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PR_REVIEW_ENSEMBLE_", env_file=".env", extra="ignore"
    )

    # Anthropic provider — routed through the local LiteLLM router by default (real Claude,
    # proxied), so no per-shell ANTHROPIC_API_KEY is needed and creds live once, server-side.
    # Set anthropic_base_url="" to use the native SDK instead (which reads ANTHROPIC_API_KEY).
    anthropic_enabled: bool = True
    anthropic_model: str = "claude-sonnet-5"
    anthropic_base_url: str = "http://localhost:4000/v1"
    anthropic_api_key: str = ""
    anthropic_max_tokens: int = 4096

    # Local LLM router (OpenAI-compatible, e.g. a LiteLLM proxy)
    local_enabled: bool = True
    local_base_url: str = "http://localhost:4000/v1"
    local_api_key: str = ""
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

    # Digest pass (single resilient pass, no fan-out). Size-guarded hybrid: a diff at/under
    # digest_max_diff_chars is digested in one shot; a larger one falls back to map-reduce —
    # split into per-file chunks, summarize each, then synthesize the digest from the summaries.
    # ~400k chars is roughly 100k input tokens, comfortably inside a large-context model.
    digest_max_diff_chars: int = 400_000
    digest_max_tokens: int = 8192  # output budget for the single-pass / reduce digest
    # Map-reduce knobs (used only when the diff is over budget).
    digest_chunk_chars: int = 100_000  # target size of each map chunk (file diffs are packed to it)
    digest_map_max_tokens: int = 2048  # output budget for each per-chunk summary
    digest_map_concurrency: int = 6  # max concurrent map calls against the router
    digest_max_chunks: int = 40  # hard cap on chunks; extras are dropped with a logged note

    # Logging
    log_path: Path = Path(__file__).parent / "logs" / "runs.jsonl"


settings = PRReviewEnsembleSettings()
