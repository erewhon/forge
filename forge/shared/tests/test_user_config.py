"""Tests for the user-level config layer (:mod:`forge.shared.user_config`)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from forge.shared import user_config as uc


@pytest.fixture
def cfg_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("FORGE_CONFIG_DIR", str(tmp_path))
    return tmp_path


def test_dir_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_CONFIG_DIR", str(tmp_path / "explicit"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert uc.user_config_dir() == tmp_path / "explicit"

    monkeypatch.delenv("FORGE_CONFIG_DIR")
    assert uc.user_config_dir() == tmp_path / "xdg" / "forge"

    monkeypatch.delenv("XDG_CONFIG_HOME")
    assert uc.user_config_dir() == Path.home() / ".config" / "forge"


def test_missing_and_malformed_config_are_non_fatal(cfg_dir: Path) -> None:
    assert uc.load_user_config() == {}  # no file
    (cfg_dir / "config.toml").write_text("this = is = not valid toml")
    assert uc.load_user_config() == {}  # malformed → empty, not an exception


def test_router_url_fans_out_only_when_unset() -> None:
    env: dict[str, str] = {"BOOK_RESEARCHER_OPENAI_BASE_URL": "http://preset/v1"}
    uc.apply_user_config({"router_url": "http://cfg/v1"}, env=env)
    # preset (higher precedence) is untouched...
    assert env["BOOK_RESEARCHER_OPENAI_BASE_URL"] == "http://preset/v1"
    # ...every other router URL var is filled from the config.
    for name in uc.ROUTER_URL_ENVS:
        assert env[name] == ("http://preset/v1" if "BOOK_RESEARCHER" in name else "http://cfg/v1")


def test_api_key_resolved_from_named_env_var_only() -> None:
    # api_key_env names a variable; its value (not the name) is applied.
    env = {"MY_ROUTER_KEY": "sk-secret"}
    uc.apply_user_config({"api_key_env": "MY_ROUTER_KEY"}, env=env)
    for name in uc.ROUTER_KEY_ENVS:
        assert env[name] == "sk-secret"
    assert "sk-secret" not in uc.ROUTER_KEY_ENVS  # sanity: we set values, not names


def test_api_key_env_unset_is_noop() -> None:
    env: dict[str, str] = {}
    uc.apply_user_config({"api_key_env": "NOT_SET_ANYWHERE"}, env=env)
    assert env == {}  # nothing to resolve → nothing written


def test_raw_env_passthrough_is_verbatim_and_respects_precedence() -> None:
    env = {"CODE_REVIEWER_MODEL": "already"}
    uc.apply_user_config(
        {"env": {"CODE_REVIEWER_MODEL": "glm", "TESTING_ENSEMBLE_OPENAI_MODEL": "coder"}},
        env=env,
    )
    assert env["CODE_REVIEWER_MODEL"] == "already"  # unset-only
    assert env["TESTING_ENSEMBLE_OPENAI_MODEL"] == "coder"


def test_key_envs_derive_from_url_envs() -> None:
    assert len(uc.ROUTER_KEY_ENVS) == len(uc.ROUTER_URL_ENVS)
    for url, key in zip(uc.ROUTER_URL_ENVS, uc.ROUTER_KEY_ENVS, strict=True):
        assert key == url.removesuffix("_BASE_URL") + "_API_KEY"


def test_router_env_lists_match_the_config_modules() -> None:
    """Drift guard: re-derive the router URL vars from the agent config modules and require the
    hard-coded :data:`ROUTER_URL_ENVS` to match exactly. A new agent (or a renamed field) fails
    here until the list is updated."""
    forge_root = Path(uc.__file__).resolve().parents[1]
    derived: set[str] = set()
    for path in forge_root.rglob("config.py"):
        text = path.read_text()
        m = re.search(r'env_prefix="([A-Z_]+)"', text)
        if not m:
            continue
        prefix = m.group(1)
        for fm in re.finditer(r"^\s{4}([a-z_]+):\s*str\s*=", text, re.M):
            field = fm.group(1)
            if field.endswith("base_url") and "zen" not in field:
                derived.add(prefix + field.upper())
    assert derived == set(uc.ROUTER_URL_ENVS)
