from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class CodeReviewerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CODE_REVIEWER_")

    # Project paths
    projects_dir: Path = Path.home() / "Projects" / "erewhon"
    repos: list[str] = [
        "agent-monitor",
        "astra",
        "finn-financial",
        "gaol",
        "hoardfs",
        "graphlib",
        "llm-router",
        "meta",
        "nous",
        "protectinator",
        "raft",
        "scrutinator",
        "steve.net",
        "tubinator",
    ]

    # Collection
    lookback_hours: int = 24
    max_diff_lines: int = 2000

    # AI backend: "anthropic" or "openai" (for local/router endpoints)
    llm_backend: str = "openai"

    # Anthropic models (used when llm_backend == "anthropic")
    anthropic_model: str = "claude-haiku-4-5-20251001"

    # OpenAI-compatible models (used when llm_backend == "openai")
    openai_base_url: str = "http://localhost:4010/v1"
    openai_api_key: str = ""
    openai_model: str = "coder"

    # Nous
    daemon_url: str = "http://127.0.0.1:7667"
    nous_data_dir: Path = Path.home() / ".local" / "share" / "nous"
    notebook_id: str = "b67b98ae-d5d2-4947-b40d-6fc6410500b6"

    # Idempotency
    review_marker: str = "<!-- nightly-code-review -->"

    # Scoring
    score_alert_threshold: int = 4  # scores at or below this trigger special attention

    # Retry: finding the daily note
    find_attempts: int = 3
    find_delay_seconds: int = 60

    @property
    def repos_paths(self) -> list[Path]:
        return [self.projects_dir / repo for repo in self.repos]


settings = CodeReviewerSettings()
