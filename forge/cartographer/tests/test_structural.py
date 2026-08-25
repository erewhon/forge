"""Structural pass over the committed mini-repo fixture (real tree-sitter parsers)."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime

import pytest

from forge.cartographer.models import MapConfig, RepoIndex, TargetInfo
from forge.cartographer.structural import build_index, render_map, write_outputs

pytest.importorskip("tree_sitter_language_pack", reason="map extra not installed")

_SYNCED = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)


def _info(config: MapConfig, i: int, stale: bool = False) -> TargetInfo:
    t = config.targets[i]
    return TargetInfo(name=t.name, host=t.host, path=t.path, synced_at=_SYNCED, stale=stale)


def _mirror(config: MapConfig, i: int) -> None:
    target = config.targets[i]
    shutil.copytree(target.path, config.mirror_dir(target), dirs_exist_ok=True)


def test_index_modules_and_symbols(config: MapConfig) -> None:
    _mirror(config, 0)
    result = build_index(config, config.targets[0], _info(config, 0))
    index = result.index

    assert {m.name for m in index.modules} == {"mini", "app", "util"}

    by_path = {f.path: f for f in index.files}
    app_js = by_path["packages/app/index.js"]
    assert app_js.module == "app"
    assert "function start" in app_js.symbols
    assert "class Server" in app_js.symbols
    assert "method Server.boot" in app_js.symbols or any(
        s.endswith("Server.boot") for s in app_js.symbols
    )
    assert any("util" in imp for imp in app_js.imports)

    tool = by_path["tool.py"]
    assert tool.module == "mini"
    assert "class Tool" in tool.symbols and any(s.endswith("Tool.run") for s in tool.symbols)
    assert tool.entry_point  # has an if __name__ == "__main__" block

    # module.files back-references stay consistent
    app_module = next(m for m in index.modules if m.name == "app")
    assert "packages/app/index.js" in app_module.files


def test_blob_sha_git_and_fallback(config: MapConfig) -> None:
    _mirror(config, 0)
    index = build_index(config, config.targets[0], _info(config, 0)).index
    by_path = {f.path: f for f in index.files}
    assert len(by_path["tool.py"].blob_sha) == 40  # tracked → git blob hash
    assert len(by_path["scratch.py"].blob_sha) == 64  # untracked → content sha256


def test_divergent_checkouts_are_independent_targets(config: MapConfig) -> None:
    for i in (0, 1):
        _mirror(config, i)
        result = build_index(config, config.targets[i], _info(config, i))
        write_outputs(config, config.targets[i], result)
    out_a = config.out_dir(config.targets[0])
    out_b = config.out_dir(config.targets[1])
    assert out_a != out_b
    assert (out_a / "index.json").is_file() and (out_b / "index.json").is_file()


def test_index_json_round_trips(config: MapConfig) -> None:
    _mirror(config, 0)
    result = build_index(config, config.targets[0], _info(config, 0))
    out = write_outputs(config, config.targets[0], result)
    loaded = RepoIndex.model_validate_json((out / "index.json").read_text())
    assert loaded.target.name == "mini-a"
    assert loaded.files and loaded.modules


def test_map_md_content_and_stale_marker(config: MapConfig) -> None:
    _mirror(config, 0)
    result = build_index(config, config.targets[0], _info(config, 0, stale=True))
    text = render_map(result)
    assert "## app" in text and "## util" in text
    assert "STALE" in text
    assert "cross-module imports" in text  # index.js imports from util


def test_oversized_files_are_skipped_with_reason(config: MapConfig) -> None:
    small = config.model_copy(update={"max_file_bytes": 10})
    _mirror(small, 0)
    result = build_index(small, small.targets[0], _info(small, 0))
    assert result.skipped, "everything exceeds 10 bytes"
    entry = next(f for f in result.index.files if f.skipped)
    assert "max_file_bytes" in entry.skipped and not entry.symbols


def test_missing_mirror_is_a_clear_error(config: MapConfig) -> None:
    with pytest.raises(FileNotFoundError, match="forge map sync"):
        build_index(config, config.targets[1], _info(config, 1))
