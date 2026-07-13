"""Disk-free preflight helper: df parsing and the fail-open threshold decision."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

from forge.task_worker import dx

_DF_OUTPUT = (
    "\x1b[2m[dx] Running: df -Pk /\x1b[0m\n"
    "Filesystem     1024-blocks      Used Available Capacity Mounted on\n"
    "/dev/loop0       209715200 190000000  19715200      91% /\n"
)


def test_parse_df_avail_skips_banner_and_header():
    # The available column is picked out past the [dx] banner and the text header.
    assert dx._parse_df_avail_kb(_DF_OUTPUT) == 19715200


def test_parse_df_avail_none_on_garbage():
    assert dx._parse_df_avail_kb("nothing numeric here\n") is None


def test_check_disk_free_blocks_when_below_floor(monkeypatch, tmp_path):
    low = _DF_OUTPUT.replace("19715200", "500000")  # ~488 MiB available
    monkeypatch.setattr(
        dx.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=low, stderr=""),
    )
    ok, status = dx.check_disk_free(tmp_path, min_free_mb=2048)
    assert not ok
    assert "container disk low" in status


def test_check_disk_free_passes_when_above_floor(monkeypatch, tmp_path):
    monkeypatch.setattr(
        dx.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=_DF_OUTPUT, stderr=""),
    )
    ok, status = dx.check_disk_free(tmp_path, min_free_mb=2048)
    assert ok
    assert "disk ok" in status


def test_check_disk_free_fails_open_on_probe_error(monkeypatch, tmp_path):
    # A flaky probe must never block all work — the gate's own diagnostics are the backstop.
    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="df", timeout=20)

    monkeypatch.setattr(dx.subprocess, "run", boom)
    ok, status = dx.check_disk_free(tmp_path, min_free_mb=2048)
    assert ok
    assert "skipped" in status
