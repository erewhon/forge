from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Gold sets are checked-in fixtures living inside this package; scorecard runs land
# next to pipeline-runs/ at the repo root (both gitignored).
_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parents[1]


class EvalsSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EVALS_")

    goldsets_dir: Path = _PACKAGE_DIR / "goldsets"
    runs_dir: Path = _REPO_ROOT / "eval-runs"
    openai_base_url: str = "http://localhost:4010/v1"
    openai_api_key: str = "sk-local-router"
    model: str = "coder"
    repeats: int = 3
    temperature: float = 0.0
    timeout: float = 240.0
    max_tokens: int = 16_000


settings = EvalsSettings()
