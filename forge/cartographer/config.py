"""Settings for the cartographer's LLM stage (the sweep/synthesis tiers).

Structural mapping needs no settings beyond ``map.toml``; everything here exists for the
summarizer. The base URL is validated by :mod:`forge.cartographer.guards` before any request —
there is no cloud tier and no fallback, by design.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from forge.shared.envfile import ENV_FILES


class CartographerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CARTOGRAPHER_", env_file=ENV_FILES, extra="ignore"
    )

    # OpenAI-compatible local router. Guarded: a non-local URL is refused at startup.
    openai_base_url: str = "http://localhost:4000/v1"
    openai_api_key: str = ""

    # Router aliases per tier. Sweep handles the high-volume file/module summaries; synthesis
    # handles the low-volume rollups and the architecture doc, where judgment matters more.
    # Sweep default is gemma for now: lightning (the preferred seat) 502s on plain completions
    # through the router's tool proxy as of 2026-08-25 — reseat via CARTOGRAPHER_SWEEP_MODEL once
    # that's fixed (LLM Router task filed).
    sweep_model: str = "gemma"
    synthesis_model: str = "ling"  # the Ling 3 router alias ("ling3" is the ensemble SEAT label)

    # Token budgets per call kind. Sized for reasoning models: the local seats (gemma, ling,
    # gpt-oss) spend a large share of the budget on reasoning tokens before any content — 700
    # was measured to yield finish_reason=length with 0 content on gemma (2026-08-25).
    file_summary_max_tokens: int = 2048
    rollup_max_tokens: int = 3072
    architecture_max_tokens: int = 4096

    # Characters of file content sent to the sweep model (files are already size-capped by the
    # structural pass; this is a second belt for pathological single-line files).
    max_prompt_chars: int = 48_000


settings = CartographerSettings()
