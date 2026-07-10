from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from forge.shared.envfile import ENV_FILES


class ParallelEditSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PARALLEL_EDIT_", env_file=ENV_FILES, extra="ignore"
    )

    # Default candidates if --models is omitted (comma-separated env override OK). Each entry is
    # "[kind:]model": a bare value or "claude:<id>" runs `claude -p`; "opencode:<ref>" runs
    # `opencode run -m <ref>` (an "llm/" prefix is added when the ref has no provider), routing
    # the open fleet (Kimi, GLM, DeepSeek, ...) through the local router.
    #
    # The default is a known-good open-fleet shortlist: all three are free (router-hosted) and
    # verified to complete inside opencode's agentic tool-loop under concurrent fan-out. This also
    # tracks the pricing direction — `claude -p` is heading toward metered, the open fleet isn't —
    # so routine fan-out defaults to open models; add a `claude-…`/`opencode:…` candidate explicitly
    # via --models when a frontier or cross-vendor comparison is wanted. (MiniMax is omitted: it's
    # healthy for plain completions but errors inside opencode's tool-loop. See 2026-06-17 log.)
    default_candidate_models: list[str] = [
        "opencode:glm-5.1",
        "opencode:kimi",
        "opencode:deepseek",
    ]

    # Candidate invocation
    claude_binary: str = "claude"
    opencode_binary: str = "opencode"
    opencode_model_prefix: str = "llm/"  # prepended to an opencode model ref lacking a provider
    per_run_timeout_seconds: float = 1800.0  # 30 min ceiling per candidate
    permission_mode: str = "acceptEdits"
    output_format: str = "text"

    # Sandbox (Ephemeral Candidate Sandboxes): when enabled, each candidate CLI
    # runs inside an ephemeral `gaol run-once` sandbox with its jj workspace
    # bind-mounted writable (plus opencode config/auth), instead of loose on the
    # host. The sandbox is created, exec'd, and destroyed per candidate (no
    # residue), with per-sandbox resource caps so concurrent fan-out is safe.
    # Off by default.
    sandbox: bool = False
    # Per-kind sandboxing: candidate kinds listed here run LOOSE on the host even when `sandbox`
    # is enabled. `claude` is trusted and authenticates via the host's OAuth (which isn't mounted
    # into the sandbox), so it stays on the host; the open-fleet kinds (opencode → router models)
    # are the ones isolated. Empty = sandbox every kind.
    sandbox_exempt_kinds: list[str] = ["claude"]
    gaol_binary: str = "gaol"
    sandbox_runtime: str = "incus"
    # Toolchain golden image built by gaol's scripts/build-candidate-image.sh
    # (a local Incus alias). Has opencode/claude/node/uv/ripgrep/fd/git so a real
    # candidate runs; cold-start is a ~1s ZFS clone. Override with a stock image
    # (e.g. "images:debian/trixie") when the toolchain isn't needed.
    sandbox_image: str = "gaol-candidate-base"
    # $HOME inside the sandbox. The candidate runs as the host workspace owner's
    # uid, which maps to this image's `dev` user; opencode reads its config here.
    sandbox_home: str = "/home/dev"
    # Per-sandbox resource caps so concurrent fan-out can't exhaust the host
    # (→ gaol run-once --memory/--cpus → Incus limits.*). None = uncapped.
    sandbox_memory: str | None = "4GiB"
    sandbox_cpus: int | None = 2
    # Mount the host's opencode config + a PER-CANDIDATE copy of its auth into
    # the sandbox so `opencode run -m llm/…` resolves the router. Per-candidate
    # state (a fresh opencode.db, only auth.json seeded) avoids the shared-sqlite
    # corruption hazard when many candidates run concurrently.
    sandbox_mount_opencode: bool = True
    # Wall-clock headroom over the per-run timeout for sandbox provision +
    # teardown; the asyncio backstop fires this much later than run-once's own
    # --timeout (which does the deterministic in-sandbox kill + reap, since a
    # SIGKILL to the gaol process would bypass its teardown guard).
    sandbox_grace_seconds: float = 120.0

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
    judge_openai_base_url: str = "http://localhost:4000/v1"
    judge_openai_api_key: str = ""
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
