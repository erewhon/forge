"""User-level forge configuration under ``~/.config/forge/``.

Forge's settings classes each read their own ``PREFIX_FIELD`` environment variables (with a repo
``.env`` layered in via :mod:`forge.shared.envfile`). This module adds a machine-level layer
*beneath* those, so a host can carry its own defaults — which LLM router to call, where the router
key lives, and any per-agent overrides — without editing the repo or exporting a long list of shell
variables. It is what you reach for when standing forge up on a new machine.

Two files, both optional:

* ``~/.config/forge/.env`` — a plain dotenv, wired into :data:`forge.shared.envfile.ENV_FILES` at
  the lowest precedence. Every settings class picks it up for free using its own variable names.
* ``~/.config/forge/config.toml`` — a friendlier, documented schema for the cross-cutting knobs.
  :func:`apply_user_config` translates it into the same environment variables the settings classes
  already read, writing each **only when it is unset** so the real environment and the repo ``.env``
  always win.

Precedence, highest first: real environment variable → repo/cwd ``.env`` → ``config.toml`` →
built-in default.

Secrets never live in ``config.toml``. ``api_key_env`` names the environment variable that holds the
router key (exported by the shell, a secrets manager, ``ho secret``, …); the key itself is resolved
at runtime and is never written to disk by forge.

Example ``config.toml``::

    router_url  = "http://router.example:4000/v1"   # OpenAI-compatible LLM router base URL
    api_key_env = "FORGE_ROUTER_API_KEY"            # name of the env var holding the router key

    [env]
    # Raw escape hatch: any PREFIX_FIELD override, verbatim. Per-agent model selection lives here,
    # since agents deliberately run different tiers:
    CODE_REVIEWER_MODEL          = "glm"
    PR_REVIEW_ENSEMBLE_LOCAL_MODEL = "coder"
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

__all__ = [
    "ROUTER_KEY_ENVS",
    "ROUTER_URL_ENVS",
    "apply_user_config",
    "load_user_config",
    "user_config_dir",
    "user_config_file",
    "user_env_file",
]

# Router endpoint / key environment variables, one pair per agent that talks to the router. These
# follow the fixed ``{PREFIX}OPENAI_BASE_URL`` / ``{PREFIX}OPENAI_API_KEY`` convention (with the two
# extra PR-review seats). Kept as data — and drift-guarded by ``tests/test_user_config.py``, which
# re-derives them from the agent config modules and fails if this list falls out of sync. The Zen
# endpoint is intentionally excluded: it is a separate, non-router backend.
ROUTER_URL_ENVS: tuple[str, ...] = (
    "BOOK_RESEARCHER_OPENAI_BASE_URL",
    "CODE_AUDIT_OPENAI_BASE_URL",
    "CODE_REVIEWER_OPENAI_BASE_URL",
    "CODING_PIPELINE_OPENAI_BASE_URL",
    "EVALS_OPENAI_BASE_URL",
    "GENERAL_RESEARCHER_OPENAI_BASE_URL",
    "PARALLEL_EDIT_JUDGE_OPENAI_BASE_URL",
    "PR_REVIEW_ENSEMBLE_ANTHROPIC_BASE_URL",
    "PR_REVIEW_ENSEMBLE_LOCAL_BASE_URL",
    "REFACTOR_ENSEMBLE_OPENAI_BASE_URL",
    "TESTING_ENSEMBLE_OPENAI_BASE_URL",
    "UPSTREAM_SYNC_OPENAI_BASE_URL",
)
ROUTER_KEY_ENVS: tuple[str, ...] = tuple(
    e[: -len("_BASE_URL")] + "_API_KEY" for e in ROUTER_URL_ENVS
)


def user_config_dir() -> Path:
    """Forge's user-config directory.

    ``$FORGE_CONFIG_DIR`` (if set) → ``$XDG_CONFIG_HOME/forge`` → ``~/.config/forge``.
    """
    override = os.environ.get("FORGE_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "forge"


def user_env_file() -> Path:
    """The user-level dotenv layered into :data:`forge.shared.envfile.ENV_FILES`."""
    return user_config_dir() / ".env"


def user_config_file() -> Path:
    """The user-level ``config.toml``."""
    return user_config_dir() / "config.toml"


def load_user_config(path: Path | None = None) -> dict[str, Any]:
    """Parse ``config.toml`` and return it as a dict. Missing or unreadable → ``{}``.

    A malformed file is not fatal: forge should still start on its built-in defaults, so a parse
    error is swallowed (and surfaced to the caller only as an empty config).
    """
    target = path or user_config_file()
    try:
        with target.open("rb") as fh:
            return tomllib.load(fh)
    except (FileNotFoundError, IsADirectoryError, tomllib.TOMLDecodeError, OSError):
        return {}


def _set_if_unset(env: MutableMapping[str, str], name: str, value: str) -> None:
    if name not in env and value != "":
        env[name] = value


def apply_user_config(
    config: dict[str, Any] | None = None,
    env: MutableMapping[str, str] | None = None,
) -> None:
    """Materialize ``config.toml`` into environment variables, each written only when unset.

    Call once at process entry (the CLI and MCP front doors do this) *before* any agent settings
    class is instantiated, so the real environment and the repo ``.env`` always take precedence.

    Recognized keys:

    * ``router_url`` — applied to every agent's router base-URL variable.
    * ``api_key_env`` — the *name* of the environment variable holding the router key; its value
      (if that variable is set) is applied to every agent's router key variable.
    * ``[env]`` — a table of ``PREFIX_FIELD = "value"`` pairs, applied verbatim.
    """
    cfg = load_user_config() if config is None else config
    target = os.environ if env is None else env

    router_url = cfg.get("router_url")
    if isinstance(router_url, str):
        for name in ROUTER_URL_ENVS:
            _set_if_unset(target, name, router_url)

    api_key_env = cfg.get("api_key_env")
    if isinstance(api_key_env, str) and api_key_env:
        key_value = target.get(api_key_env, "")
        if key_value:
            for name in ROUTER_KEY_ENVS:
                _set_if_unset(target, name, key_value)

    raw = cfg.get("env")
    if isinstance(raw, dict):
        for name, value in raw.items():
            if isinstance(name, str) and value is not None:
                _set_if_unset(target, name, str(value))
