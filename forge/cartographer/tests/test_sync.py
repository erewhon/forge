"""Sync behavior with an injected runner — no rsync/ssh processes are ever spawned."""

from __future__ import annotations

import subprocess

from forge.cartographer.models import MapConfig
from forge.cartographer.sync import _rsync_command, read_sync_detail, read_sync_state, sync_target


def _proc(returncode: int, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="", stderr=stderr)


def test_rsync_command_shape(config: MapConfig) -> None:
    target = config.targets[0]
    cmd = _rsync_command(target, config.mirror_dir(target))
    assert cmd[0] == "rsync"
    assert "--delete" in cmd
    assert any(arg.startswith("ssh -o BatchMode=yes") for arg in cmd)
    assert cmd[-2] == f"vm-a.test:{target.path}/"
    assert "--exclude" in cmd and "node_modules" in cmd
    assert ".git" not in cmd  # .git must transfer: blob hashes come from the mirror's index


def test_sync_success_records_state(config: MapConfig) -> None:
    target = config.targets[0]
    result = sync_target(config, target, runner=lambda cmd: _proc(0))
    assert result.ok
    synced_at, stale = read_sync_state(config, target)
    assert synced_at is not None and not stale


def test_sync_failure_keeps_previous_timestamp_and_marks_stale(config: MapConfig) -> None:
    target = config.targets[0]
    sync_target(config, target, runner=lambda cmd: _proc(0))
    first_synced, _ = read_sync_state(config, target)

    result = sync_target(config, target, runner=lambda cmd: _proc(255, "ssh: unreachable\n"))
    assert not result.ok and "unreachable" in result.message
    synced_at, stale = read_sync_state(config, target)
    assert stale and synced_at == first_synced


def test_unsynced_target_reads_as_stale(config: MapConfig) -> None:
    synced_at, stale = read_sync_state(config, config.targets[1])
    assert synced_at is None and stale


def test_one_unreachable_target_does_not_block_the_other(config: MapConfig) -> None:
    down = {config.targets[0].key}

    def runner(cmd: list[str]) -> subprocess.CompletedProcess:
        return _proc(255, "down") if any(k in " ".join(cmd) for k in down) else _proc(0)

    results = [sync_target(config, t, runner=runner) for t in config.targets]
    assert [r.ok for r in results] == [False, True]


def test_target_selection_unknown_name(config: MapConfig) -> None:
    try:
        config.target_selection(["nope"])
    except KeyError as exc:
        assert "nope" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected KeyError")


def test_target_keys_differ_per_host(config: MapConfig) -> None:
    a, b = config.targets
    assert a.key != b.key  # same path, different VM → different checkouts


def test_sync_returncode_recorded(config: MapConfig) -> None:
    """Returncode is captured in SyncResult."""
    target = config.targets[0]
    result = sync_target(config, target, runner=lambda cmd: _proc(42))
    assert result.returncode == 42
    assert not result.ok


def test_sync_stderr_tail_recorded(config: MapConfig) -> None:
    """stderr tail (last 5 lines) is captured in SyncResult."""
    target = config.targets[0]
    stderr_input = "line1\nline2\nline3\nline4\nline5\nline6\nline7\n"
    result = sync_target(config, target, runner=lambda cmd: _proc(1, stderr_input))
    assert result.stderr_tail == "\n".join(stderr_input.strip().splitlines()[-5:])


def test_sync_consecutive_failures_counter(config: MapConfig) -> None:
    """consecutive_failures increments on failure and resets on success."""
    target = config.targets[0]

    # First failure
    result = sync_target(config, target, runner=lambda cmd: _proc(1, "error1\n"))
    assert not result.ok
    detail = read_sync_detail(config, target)
    assert detail.consecutive_failures == 1

    # Second failure - counter should increment
    result = sync_target(config, target, runner=lambda cmd: _proc(1, "error2\n"))
    assert not result.ok
    detail = read_sync_detail(config, target)
    assert detail.consecutive_failures == 2

    # Success - counter should reset to 0
    result = sync_target(config, target, runner=lambda cmd: _proc(0))
    assert result.ok
    detail = read_sync_detail(config, target)
    assert detail.consecutive_failures == 0
