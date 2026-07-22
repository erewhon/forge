"""Config loading + env-level defaults for `forge grind`.

The grind config is a YAML/JSON file validated into :class:`GrindConfig` (mirrors
``book_researcher``). Machine-level defaults (the model, the iteration cap) come from the
environment so a work machine can set them once without editing every runbook.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

from forge.grind.models import GrindConfig


class Settings(BaseSettings):
    """Env-level grind defaults. A config's own value (or a CLI flag) wins over these."""

    model_config = SettingsConfigDict(env_prefix="GRIND_", extra="ignore")

    #: Default OpenCode model string when neither the config nor --model sets one.
    model: str = "opencode/anthropic/claude-sonnet-4"
    #: Fallback iteration cap when the config omits max_iterations.
    max_iterations: int = 20


settings = Settings()


def load_config(config_path: str | Path) -> GrindConfig:
    """Load and validate a grind runbook from a YAML or JSON file."""
    path = Path(config_path).expanduser().resolve()
    text = path.read_text()
    data = yaml.safe_load(text) if path.suffix in (".yaml", ".yml") else json.loads(text)
    return GrindConfig.model_validate(data)


def resolve_model(config: GrindConfig, cli_model: str | None) -> str:
    """The OpenCode model string, precedence: --model > config.model > GRIND_MODEL default."""
    return cli_model or config.model or settings.model
