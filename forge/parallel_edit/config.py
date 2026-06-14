from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class ParallelEditSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PARALLEL_EDIT_")

    # Default candidates if --models is omitted (comma-separated env override OK). Each entry is
    # "[kind:]model": a bare value or "claude:<id>" runs `claude -p`; "opencode:<ref>" runs
    # `opencode run -m <ref>` (an "llm/" prefix is added when the ref has no provider), routing
    # the open fleet (Kimi, Qwen, GLM, MiniMax, ...) through the local router.
    default_candidate_models: list[str] = [
        "claude-opus-4-8",
        "claude-sonnet-4-6",
    ]

    # Candidate invocation
    claude_binary: str = "claude"
    opencode_binary: str = "opencode"
    opencode_model_prefix: str = "llm/"  # prepended to an opencode model ref lacking a provider
    per_run_timeout_seconds: float = 1800.0  # 30 min ceiling per candidate
    permission_mode: str = "acceptEdits"
    output_format: str = "text"

    # Judge LLM backend: "openai" (local LLM router) or "anthropic".
    # Defaults to the router because this environment authenticates Claude through
    # Claude Code's OAuth (no ANTHROPIC_API_KEY for the raw SDK): candidates run via
    # `claude` and inherit that auth, but the judge calls an SDK directly and needs
    # its own key. The router has one, hosts capable non-Claude models (which also
    # avoids judge/candidate self-preference), and matches the rest of the fleet.
    # Set judge_backend="anthropic" when ANTHROPIC_API_KEY is available.
    judge_backend: str = "openai"

    # Anthropic judge
    judge_anthropic_model: str = "claude-opus-4-7"
    judge_anthropic_max_tokens: int = 8192

    # OpenAI-compatible judge (e.g. local LiteLLM router)
    judge_openai_base_url: str = "http://localhost:4010/v1"
    judge_openai_api_key: str = "sk-local-router"
    judge_openai_model: str = "coder"
    judge_openai_max_tokens: int = 8192

    # Judge resilience: the judge runs through the ensemble harness's failover Pool.
    # The primary is judge_backend/judge_*_model above; these router models are appended
    # after it and only used if the primary fails (pulled / auth / timeout). All failover
    # members go through the router because it always has a key — so even an Anthropic
    # primary degrades to a reachable open model rather than aborting the run.
    judge_failover_models: list[str] = ["qwen3.6-plus"]
    judge_timeout_seconds: float = 120.0
    # Local judge models are non-deterministic about JSON formatting: a call can succeed yet
    # return unparseable output. The judge pool validates the verdict parses and treats a failure
    # as transient, so it re-rolls on the same model this many times before failing over.
    judge_max_attempts_per_model: int = 3

    # Workspace management
    workspace_base_dir: Path | None = None  # default: repo's parent dir
    workspace_name_prefix: str = "pe"
    cleanup_on_success: bool = True
    cleanup_on_failure: bool = False  # keep failed workspaces for inspection

    # Truncation when stuffing into the judge prompt
    max_diff_chars_per_candidate: int = 120_000

    # Logging
    log_path: Path = Path(__file__).parent / "logs" / "runs.jsonl"


settings = ParallelEditSettings()
