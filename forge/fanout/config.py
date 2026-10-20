from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from forge.shared.envfile import ENV_FILES


class FanoutSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FANOUT_", env_file=ENV_FILES, extra="ignore")

    # Default seats if --models is omitted. Each entry is "[kind:]model": a bare value or
    # "claude:<id>" runs `claude -p`; "pi:<id>" runs `pi -p --provider <pi_provider> --model <id>`.
    # The pi router token is scoped to fleet-local models, so pi seats must name models that
    # are actually on fleet hardware (role aliases like `coder`, or placements like
    # `gemma4-26b`); cloud-backed names such as `qwen3.6-plus` come back as a router 403 and
    # the seat ends up `error`. Probe one with
    # `pi -p --no-tools --provider llm-router --model <id> "Reply OK"` before a long sweep.
    # Verified 2026-10-20: gemma4-26b completes the sweep (~11 min, structured findings);
    # glm-5.3-flash (the `coder` role) loops on tool calls for 140+ turns and never answers.
    default_models: list[str] = ["claude:opus", "pi:gemma4-26b"]

    # Seat invocation
    claude_binary: str = "claude"
    pi_binary: str = "pi"
    pi_provider: str = "llm-router"
    per_seat_timeout_seconds: float = 1800.0  # 30 min ceiling per seat

    # Read-only tool allowlists. Analysis seats explore but never edit: claude gets the read
    # tools plus ripgrep; everything else (Edit/Write) is denied outright in -p mode. pi's
    # built-in read-only set is read/grep/find/ls (no bash).
    claude_allowed_tools: list[str] = ["Read", "Grep", "Glob", "Bash(rg:*)"]
    claude_disallowed_tools: list[str] = ["Edit", "Write", "NotebookEdit"]
    pi_tools: list[str] = ["read", "grep", "find", "ls"]
    # The owner's pi `guardrails` extension caps tool calls per headless turn at 12 and then
    # refuses every tool, which leaves an exploring seat with nothing to say. An analysis sweep
    # needs dozens of reads, so each pi seat gets this budget via PI_GUARD_BUDGET (harmless where
    # the extension is absent). Kept finite: a model that loops (glm-5.3-flash ran 146 turns
    # under a 200 budget) gets told to answer with what it has instead of burning the timeout.
    pi_tool_budget: int = 60

    # Judge: an interactive `claude` session opened from the generated judge prompt.
    judge_model: str = "opus"

    # Where sweeps land when --out is omitted: <out_base>/<repo-name>-<YYYYmmdd-HHMMSS>
    out_base: Path = Path.home() / "fanout"


settings = FanoutSettings()
