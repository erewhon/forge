from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from forge.shared.envfile import ENV_FILES


class PRReviewEnsembleSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PR_REVIEW_ENSEMBLE_", env_file=ENV_FILES, extra="ignore"
    )

    # Anthropic provider — routed through the local LiteLLM router by default (real Claude,
    # proxied), so no per-shell ANTHROPIC_API_KEY is needed and creds live once, server-side.
    # Set anthropic_base_url="" to use the native SDK instead (which reads ANTHROPIC_API_KEY).
    anthropic_enabled: bool = True
    anthropic_model: str = "claude-sonnet-5"
    anthropic_base_url: str = "http://localhost:4000/v1"
    anthropic_api_key: str = ""
    anthropic_max_tokens: int = 4096

    # The local LLM router (OpenAI-compatible LiteLLM proxy) — now the shared endpoint for the
    # WHOLE roster (glm/minimax/m3/kimi/coder all resolve here, and sonnet is proxied too), so
    # creds live once, server-side. .env points this at localhost:4010.
    local_enabled: bool = True
    local_base_url: str = "http://localhost:4000/v1"
    local_api_key: str = ""
    local_model: str = "coder"  # (legacy; the roster now names models directly in providers.py)
    local_max_tokens: int = 4096

    # Legacy OpenCode-Zen-direct fields — no longer used: zen models (glm/m3/kimi) now ride the
    # router above, so no separate Zen endpoint/key is needed. Kept for back-compat with old .env.
    opencode_zen_enabled: bool = True
    opencode_zen_base_url: str = "https://opencode.ai/zen/v1"
    opencode_zen_api_key: str = ""
    opencode_zen_model: str = "kimi-k2.6"
    opencode_zen_max_tokens: int = 4096

    # Aggregator: preferred synthesizer, tried first. The aggregator runs through a failover
    # Pool whose rotation is [preferred, then ROTATION_ORDER]; if every member is down,
    # AggregateCombiner falls back to deterministic concatenation. Sonnet is the strongest
    # synthesizer, so it leads by default.
    aggregator_provider: str = "sonnet-5"
    aggregator_max_tokens: int = 4096

    # Runner. Both budgets resized 2026-07-31 for the local minimax seat (M2.7-REAP on
    # archimedes): a reasoning model at ~26 tok/s spends thousands of tokens thinking before
    # the review — at 4096 its output truncated (same failure the epic-gate verifier hit with
    # glm and fixed by raising to 16k; a ceiling, not a reservation, so slack costs nothing),
    # and at 120s a full review would routinely blow the timeout and fail over to Zen m3,
    # quietly turning the local seat into a cloud seat that also wastes 120s per run.
    per_provider_timeout_seconds: float = 300.0
    review_max_tokens: int = (
        16384  # max_tokens for each reviewer's pass (Prompt-level in the harness)
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
