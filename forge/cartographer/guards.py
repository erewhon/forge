"""The cartographer's disclosure guards.

Contract code may only ever reach **local** models, and everything derived from it (summaries,
maps, caches) must stay on the home network. Both rules are enforced here, in code, with no
escape hatch — a config knob that could widen them would defeat their purpose.

The host test is textual and deterministic (no DNS): private/loopback/CGNAT IP literals,
bare LAN hostnames, and the home overlay domains. A public name that happens to resolve
privately is still refused — err on the side of refusing.
"""

from __future__ import annotations

import ipaddress
import re
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

#: Overlay/home DNS suffixes that count as the home network.
_PRIVATE_SUFFIXES = (".bcc.sh", ".ts.net", ".internal", ".lan", ".home.arpa")


class DisclosureError(SystemExit):
    """Raised (as a process-exiting error) when a guard refuses to proceed."""


def _host_of(url_or_remote: str) -> str:
    """Best-effort host extraction from a URL or a git remote spec."""
    if "://" in url_or_remote:
        return urlsplit(url_or_remote).hostname or ""
    # scp-like git remote: [user@]host:path
    m = re.match(r"^(?:[^@/\s]+@)?([^:/\s]+):", url_or_remote)
    if m:
        return m.group(1)
    return ""  # bare path → local filesystem, no host


def is_home_host(host: str) -> bool:
    if not host:
        return True  # no host at all: a local path
    host = host.strip("[]").lower()
    try:
        ip = ipaddress.ip_address(host)
        return (
            ip.is_private  # RFC1918 / ULA / loopback / link-local
            or ip in ipaddress.ip_network("100.64.0.0/10")  # CGNAT — the tailnet range
        )
    except ValueError:
        pass
    if host == "localhost" or "." not in host:
        return True  # bare LAN/mesh name (talos, code-mesh, work-cfa…)
    return host.endswith(_PRIVATE_SUFFIXES)


def guard_router_url(base_url: str) -> None:
    """Guard 1: the inference endpoint must be on the home network."""
    host = _host_of(base_url)
    if not is_home_host(host):
        raise DisclosureError(
            f"forge map: refusing router URL {base_url!r} — contract code may only reach "
            "local models; there is no cloud tier and no fallback."
        )


def _git_remotes(worktree: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(worktree), "config", "--get-regexp", r"^remote\..*\.url$"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return []
    return [line.split(maxsplit=1)[1] for line in proc.stdout.splitlines() if " " in line]


def _enclosing_repos(path: Path) -> list[Path]:
    """Git dirs whose work tree contains ``path`` — including a pure (non-colocated) jj store."""
    repos: list[Path] = []
    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if proc.returncode == 0 and proc.stdout.strip():
        repos.append(Path(proc.stdout.strip()))
    for parent in (path, *path.parents):
        jj_git = parent / ".jj" / "repo" / "store" / "git"
        if jj_git.is_dir():
            repos.append(parent / ".jj" / "repo" / "store" / "git")
    return repos


def guard_output_root(output_root: Path) -> None:
    """Guard 2: derived artifacts must not live where a public remote could publish them."""
    existing = output_root
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    for repo in _enclosing_repos(existing):
        for remote in _git_remotes(repo):
            host = _host_of(remote)
            if not is_home_host(host):
                raise DisclosureError(
                    f"forge map: refusing output_root {output_root} — it sits inside a "
                    f"work tree with a public remote ({remote}); derived maps are "
                    "contract-derived and must stay on the home network."
                )


def run_guards(base_url: str, output_root: Path) -> None:
    """Both guards, in order. Called before any inference — including by later stages."""
    guard_router_url(base_url)
    guard_output_root(output_root)
