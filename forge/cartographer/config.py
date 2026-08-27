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
    # (The 2026-08-25 lightning-502-via-tool-proxy bug is fixed; a request that exhausts
    # max_tokens while the model is still reasoning now returns 200 with empty content and
    # finish_reason "length" — keep the budgets below above the model's reasoning appetite.)
    sweep_model: str = "lightning"
    synthesis_model: str = "ling"  # the Ling 3 router alias ("ling3" is the ensemble SEAT label)

    # Token budgets per call kind. Sized for reasoning models: the local seats (gemma, ling,
    # gpt-oss) spend a large share of the budget on reasoning tokens before any content — 700
    # was measured to yield finish_reason=length with 0 content on gemma (2026-08-25).
    file_summary_max_tokens: int = 2048
    rollup_max_tokens: int = 3072
    architecture_max_tokens: int = 4096
    # Condense (interim-digest) calls read a near-budget prompt, and the sweep seat's
    # reasoning appetite scales with prompt size: 2048 was measured EMPTY on lightning for a
    # 47k-char condense prompt, 3072 returned content (2026-08-28). 4096 leaves the digest
    # itself some room after reasoning.
    condense_max_tokens: int = 4096

    # Characters of file content sent to the sweep model (files are already size-capped by the
    # structural pass; this is a second belt for pathological single-line files).
    max_prompt_chars: int = 48_000

    # Ceiling on a synthesis prompt, in characters (~4 chars/token). Bigger rollup/architecture
    # inputs are reduced first: batches of sections → interim digests on the SWEEP seat → one
    # final pass on the synthesis seat, repeating if the digests still don't fit. 48k chars is
    # the sweep envelope already proven by max_prompt_chars; the synthesis seat 502'd on ~24k-
    # token prefills (2026-08-27), so don't raise this without re-measuring both seats.
    # (A 1,400-file module rollup measured 330k prompt tokens — module size is unbounded, so
    # synthesis must be bounded here.)
    synthesis_prompt_budget_chars: int = 48_000

    # Per-request wall-clock ceiling. Near-budget synthesis prompts prefill slowly on the local
    # seats — a ~24k-token condense prompt blew the OpenAI SDK's 600s default on ling
    # (2026-08-27); steady-state file summaries never get close to this.
    llm_timeout_seconds: float = 1800.0


settings = CartographerSettings()
