"""Locate the machine-local .env regardless of the caller's working directory.

Settings classes point ``env_file`` here so an editable install reads the repo-root ``.env``
from any cwd (``__file__`` → ``forge/shared/`` → repo root). A ``.env`` in the caller's cwd is
listed second, so project-local values override the repo-wide ones. For non-editable installs
the repo-root path lands inside site-packages and simply doesn't exist — pydantic-settings
skips missing env files silently.
"""

from __future__ import annotations

from pathlib import Path

ENV_FILES = (Path(__file__).resolve().parents[2] / ".env", Path(".env"))
