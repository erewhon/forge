"""Locate the machine-local .env regardless of the caller's working directory.

Settings classes point ``env_file`` here so an editable install reads the repo-root ``.env``
from any cwd (``__file__`` → ``forge/shared/`` → repo root). A ``.env`` in the caller's cwd is
listed last, so project-local values override the repo-wide ones. For non-editable installs
the repo-root path lands inside site-packages and simply doesn't exist — pydantic-settings
skips missing env files silently.

The user-level ``~/.config/forge/.env`` is listed *first* (lowest precedence): it provides
machine-wide defaults that the repo and cwd ``.env`` — and the real environment — override. See
:mod:`forge.shared.user_config` for the richer ``config.toml`` layer that sits above it.
"""

from __future__ import annotations

from pathlib import Path

from forge.shared.user_config import user_env_file

ENV_FILES = (user_env_file(), Path(__file__).resolve().parents[2] / ".env", Path(".env"))
