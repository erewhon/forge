from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from forge.shared.envfile import ENV_FILES


class CodeReviewerSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CODE_REVIEWER_", env_file=ENV_FILES, extra="ignore"
    )

    # Project paths. `repos` is the list of checkout names under projects_dir to review;
    # set it via CODE_REVIEWER_REPOS as a JSON list (e.g. '["repo-a","repo-b"]').
    projects_dir: Path = Path.home() / "projects"
    repos: list[str] = []

    # Collection
    lookback_hours: int = 24
    max_diff_lines: int = 2000

    # AI backend: "anthropic" or "openai" (for local/router endpoints)
    llm_backend: str = "openai"

    # Anthropic models (used when llm_backend == "anthropic")
    anthropic_model: str = "claude-haiku-4-5-20251001"

    # OpenAI-compatible models (used when llm_backend == "openai")
    openai_base_url: str = "http://localhost:4000/v1"
    openai_api_key: str = ""
    openai_model: str = "coder"

    # Nous daily-note sink — off by default so a plain install never touches Nous. Enabling it
    # (CODE_REVIEWER_NOUS_SINK=1) needs a reachable Nous daemon; the daemon settings below only
    # matter then.
    nous_sink: bool = False
    daemon_url: str = "http://127.0.0.1:7667"
    nous_data_dir: Path = Path.home() / ".local" / "share" / "nous"
    # UUID of the notebook receiving daily-note reviews; required when nous_sink is on
    # (set CODE_REVIEWER_NOTEBOOK_ID).
    notebook_id: str = ""

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
