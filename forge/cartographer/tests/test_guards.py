"""Both disclosure guards, both directions — messages asserted, not just exit codes, so an
inverted guard cannot pass."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from forge.cartographer.guards import (
    DisclosureError,
    guard_output_root,
    guard_router_url,
    is_home_host,
)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:4000/v1",
        "http://127.0.0.1:4010/v1",
        "http://talos:4010/v1",  # bare LAN name
        "http://192.168.42.7:4010/v1",
        "http://10.0.0.5:4000/v1",
        "http://100.83.1.2:4010/v1",  # CGNAT / tailnet range
        "http://router.m.bcc.sh/v1",
        "https://llm.peacock-bramble.ts.net/v1",
    ],
)
def test_router_guard_allows_home_hosts(url: str) -> None:
    guard_router_url(url)  # must not raise


@pytest.mark.parametrize(
    "url",
    [
        "https://api.openai.com/v1",
        "https://api.anthropic.com",
        "http://8.8.8.8:4000/v1",
        "https://my-router.example.com/v1",
    ],
)
def test_router_guard_refuses_public_hosts(url: str) -> None:
    with pytest.raises(DisclosureError, match="only reach local models"):
        guard_router_url(url)


def test_no_host_is_home() -> None:
    assert is_home_host("")


def _repo_with_remote(root: Path, remote: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(root), "remote", "add", "origin", remote],
        check=True,
        capture_output=True,
    )
    return root


def test_output_guard_refuses_public_remote(tmp_path: Path) -> None:
    repo = _repo_with_remote(tmp_path / "repo", "git@github.com:someone/notes.git")
    inside = repo / "maps"
    with pytest.raises(DisclosureError, match="stay on the home network"):
        guard_output_root(inside)


def test_output_guard_allows_home_remote_and_plain_dirs(tmp_path: Path) -> None:
    repo = _repo_with_remote(tmp_path / "repo", "ssh://code-mesh:23231/erewhon/maps")
    guard_output_root(repo / "maps")  # home git server → fine
    guard_output_root(tmp_path / "no-repo" / "maps")  # not a work tree at all → fine


def test_output_guard_checks_nonexistent_root_via_parents(tmp_path: Path) -> None:
    repo = _repo_with_remote(tmp_path / "repo", "https://github.com/someone/notes")
    with pytest.raises(DisclosureError, match="stay on the home network"):
        guard_output_root(repo / "not" / "created" / "yet")
