from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SweepSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SWEEP_")

    # SSH destination of the Soft Serve instance (e.g. "code-public") — required.
    host: str = ""
    port: int = 23231

    # Where the sweep keeps its clones. Machine-owned cache: a refresh hard-resets each
    # clone to origin, so nothing of value may live only here.
    workdir: Path = Path.home() / ".cache" / "forge-sweep"

    # fnmatch globs on the repo names `repo list` returns (e.g. "natorinator/*").
    include: list[str] = Field(default_factory=lambda: ["*"])
    exclude: list[str] = Field(default_factory=list)

    # repo name -> upstream URL. Presence enables `forge upstream` for that repo (a fresh
    # clone has no upstream remote — the fork relationship is config, not clonable state).
    # Env form is JSON: SWEEP_UPSTREAM_REMOTES='{"me/fork": "https://github.com/up/stream"}'
    upstream_remotes: dict[str, str] = Field(default_factory=dict)

    deps_enabled: bool = True
    upstream_enabled: bool = True

    # Task-store env injected into each agent run. "" = inherit the caller's env untouched.
    # git-bug is the default: advisories land IN the swept repo and travel with it.
    task_store_backend: str = "git-bug"
    bug_user_name: str = "forge-sweep"
    bug_user_email: str = "forge-sweep@localhost"

    ssh_timeout: int = 60
    clone_timeout: int = 600
    run_timeout: int = 2400  # matches the bumper's systemd TimeoutStartSec

    auto_log_path: Path = Path(__file__).parent / "logs" / "sweep.jsonl"


settings = SweepSettings()
