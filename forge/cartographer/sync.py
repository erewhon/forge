"""Mirror configured checkouts into ``<output_root>/mirror/<key>`` via rsync-over-SSH.

An unreachable host degrades, never aborts: the target keeps its previous mirror (marked stale
in a sidecar state file) and the structural pass still runs over whatever is on disk.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from forge.cartographer.models import DEFAULT_IGNORES, MapConfig, TargetConfig

_SSH = "ssh -o BatchMode=yes -o ConnectTimeout=8"

#: Injectable process runner so tests never spawn rsync.
Runner = Callable[[list[str]], subprocess.CompletedProcess]


@dataclass(frozen=True)
class SyncResult:
    key: str
    ok: bool
    returncode: int | None = None
    message: str = ""
    stderr_tail: str = ""


@dataclass
class SyncDetail:
    synced_at: datetime | None
    stale: bool
    consecutive_failures: int


def _default_runner(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def _rsync_command(target: TargetConfig, mirror: Path) -> list[str]:
    base = target.path.rstrip("/")
    src = f"{target.host}:{base}/" if target.host else f"{base}/"
    cmd = ["rsync", "-a", "--delete", "-e", _SSH]
    for glob in (*DEFAULT_IGNORES, *target.ignore_globs):
        # .git stays: the structural pass reads blob hashes from the mirror's git index.
        if glob in (".git", ".jj"):
            continue
        cmd += ["--exclude", glob]
    cmd += [src, f"{mirror}/"]
    return cmd


def _state_path(config: MapConfig, target: TargetConfig) -> Path:
    return config.output_root / "mirror" / f"{target.key}.sync.json"


def read_sync_state(config: MapConfig, target: TargetConfig) -> tuple[datetime | None, bool]:
    """(synced_at, stale) for a target. No state file → never synced → stale."""
    try:
        raw = json.loads(_state_path(config, target).read_text())
        synced_at = datetime.fromisoformat(raw["synced_at"]) if raw.get("synced_at") else None
        return synced_at, not raw.get("ok", False)
    except (OSError, ValueError, KeyError):
        return None, True


def read_sync_detail(config: MapConfig, target: TargetConfig) -> SyncDetail:
    """Return synced_at, stale, and consecutive_failures for a target."""
    try:
        raw = json.loads(_state_path(config, target).read_text())
        synced_at = datetime.fromisoformat(raw["synced_at"]) if raw.get("synced_at") else None
        stale = not raw.get("ok", False)
        consecutive_failures = raw.get("consecutive_failures", 0)
        return SyncDetail(
            synced_at=synced_at,
            stale=stale,
            consecutive_failures=consecutive_failures,
        )
    except (OSError, ValueError, KeyError):
        return SyncDetail(synced_at=None, stale=True, consecutive_failures=0)


def sync_target(
    config: MapConfig, target: TargetConfig, runner: Runner = _default_runner
) -> SyncResult:
    mirror = config.mirror_dir(target)
    mirror.mkdir(parents=True, exist_ok=True)
    proc = runner(_rsync_command(target, mirror))
    ok = proc.returncode == 0
    state = _state_path(config, target)
    state.parent.mkdir(parents=True, exist_ok=True)
    detail = read_sync_detail(config, target)
    consecutive_failures = (detail.consecutive_failures + 1) if not ok else 0
    state.write_text(
        json.dumps(
            {
                # A failed sync keeps the previous timestamp: the mirror content is that old.
                "synced_at": datetime.now(UTC).isoformat() if ok else
                (detail.synced_at.isoformat() if detail.synced_at else None),
                "ok": ok,
                "consecutive_failures": consecutive_failures,
            }
        )
    )
    stderr_lines = (proc.stderr or "").splitlines()
    stderr_tail = "\n".join(stderr_lines[-5:]) if stderr_lines else ""
    message = "" if ok else (proc.stderr or proc.stdout or "rsync failed").strip().splitlines()[-1]
    return SyncResult(
        key=target.key,
        ok=ok,
        returncode=proc.returncode,
        message=message,
        stderr_tail=stderr_tail,
    )


def sync_all(
    config: MapConfig, targets: list[TargetConfig], runner: Runner = _default_runner
) -> list[SyncResult]:
    return [sync_target(config, t, runner) for t in targets]
