from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class UpstreamSyncSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="UPSTREAM_SYNC_")

    remote: str = "upstream"
    # "" auto-detects (main, then master) — on the remote and locally, respectively.
    upstream_branch: str = ""
    local_branch: str = ""
    branch_prefix: str = "upstream-sync"

    fetch_timeout: int = 300
    merge_timeout: int = 300

    # The collision seat (LLM). Disabled -> verdict is None, which blocks --auto-merge
    # (fail-closed) but never the default branch push.
    seat_enabled: bool = True
    seat_model: str = "coder"
    seat_max_tokens: int = 4096
    openai_base_url: str = "http://localhost:4000/v1"
    openai_api_key: str = ""
    # Full hunks are shown only for overlap files, capped; the rest of the upstream change
    # arrives as a complete --stat manifest (diff-literacy: absence from hunks != unchanged).
    diff_cap: int = 20000
    log_cap: int = 100  # upstream commits shown to the seat and the advisory task

    auto_log_path: Path = Path(__file__).parent / "logs" / "upstream.jsonl"


settings = UpstreamSyncSettings()
