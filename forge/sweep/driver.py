"""The sweep loop: enumerate → clone/refresh → run agents per repo → summarize.

A thin driver over existing machinery — no gates and no landing logic live here; those
belong to the per-repo agents (`forge deps`, `forge upstream`). What the driver owns:

- **Enumeration** — Soft Serve's SSH CLI (``ssh -p <port> <host> repo list``, one bare
  repo name per line; live-verified 2026-07-12).
- **A machine-owned workdir of clones** — clone on first sight, ``fetch`` +
  ``reset --hard origin/<default>`` on later sweeps. The reset is deliberate: the workdir
  is a cache, and anything of value was pushed by the agent that produced it.
- **Per-repo env injection** — each agent runs as a subprocess with
  ``TASK_STORE_BACKEND``/``GIT_BUG_TASK_STORE_*`` set for THAT clone. Subprocesses are
  what make this correct: the task-store settings are import-time singletons, so mutating
  ``os.environ`` in-process would not re-point them (and a crash stays contained).
- **Fail isolation** — one repo failing never stops the sweep. The sweep exits non-zero
  only for driver-level failures (no host, enumeration failed, workdir unusable).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path

from forge.dependabot.ecosystems import EcosystemError, detect_ecosystem
from forge.shared.automerge import log_decision
from forge.shared.gitops import GitError, detect_branch, git, git_ok
from forge.sweep.config import settings
from forge.sweep.models import AgentRun, SweepResult


class SweepError(RuntimeError):
    """A driver-level failure: enumeration or workdir, not any single repo."""


# The agents' rendered summaries carry their status headline; the LAST one wins (deps
# may log interim lines first).
_STATUS_RE = re.compile(r"^# (?:meta deps|forge upstream) — ([\w-]+)", re.MULTILINE)

_AGENT_MODULES = {
    "deps": "forge.dependabot.main",
    "upstream": "forge.upstream_sync.main",
}


def list_repos(host: str, port: int, *, timeout: int) -> list[str]:
    """Repo names on the instance, one per line from ``repo list``."""
    result = subprocess.run(
        ["ssh", "-p", str(port), host, "repo", "list"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise SweepError(
            f"ssh {host} repo list failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def filter_repos(names: list[str], include: list[str], exclude: list[str]) -> list[str]:
    return [
        n
        for n in names
        if any(fnmatch(n, g) for g in include) and not any(fnmatch(n, g) for g in exclude)
    ]


def clone_url(host: str, port: int, name: str) -> str:
    return f"ssh://{host}:{port}/{name}"


def ensure_clone(url: str, dest: Path, *, timeout: int) -> Path:
    """Clone on first sight; otherwise fetch and hard-reset to origin's default branch."""
    if not (dest / ".git").exists():
        result = subprocess.run(
            ["git", "clone", url, str(dest)], capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            raise GitError(f"git clone {url} failed: {result.stderr.strip()}")
        return dest
    git(dest, "fetch", "origin", timeout=timeout)
    branch = detect_branch(dest, "refs/remotes/origin")
    git(dest, "checkout", branch)
    git(dest, "reset", "--hard", f"origin/{branch}")
    return dest


def ensure_upstream_remote(clone: Path, url: str) -> None:
    """Wire the fork's upstream remote into the clone (config, not clonable state)."""
    if git_ok(clone, "remote", "get-url", "upstream"):
        if git(clone, "remote", "get-url", "upstream") != url:
            git(clone, "remote", "set-url", "upstream", url)
    else:
        git(clone, "remote", "add", "upstream", url)


def ensure_bug_identity(clone: Path, *, name: str, email: str) -> None:
    """git-bug needs an identity per repo before it can file bugs; create one once."""
    probe = subprocess.run(
        ["git-bug", "user"], cwd=str(clone), capture_output=True, text=True, timeout=60
    )
    if probe.returncode == 0:
        return
    subprocess.run(
        ["git-bug", "user", "new", "--non-interactive", "-n", name, "-e", email],
        cwd=str(clone),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )


def bug_sync(clone: Path, direction: str, *, timeout: int) -> str | None:
    """``git-bug pull``/``push`` — best-effort; returns a warning string on failure.

    Pull keeps advisory dedupe accurate across machines; push publishes filed advisories
    so they render in the sprinkles UI. Neither may fail the sweep."""
    try:
        result = subprocess.run(
            ["git-bug", direction],
            cwd=str(clone),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return f"git-bug {direction}: {e}"
    if result.returncode != 0:
        return f"git-bug {direction} failed: {result.stderr.strip()[:200]}"
    return None


def _parse_status(stdout: str, returncode: int) -> str:
    found = _STATUS_RE.findall(stdout)
    if found:
        return found[-1]
    return "ok" if returncode == 0 else "error"


def run_agent(
    repo_name: str,
    clone: Path,
    agent: str,
    *,
    project: str,
    dry_run: bool,
    auto_merge: bool,
    backend: str,
    timeout: int,
) -> AgentRun:
    """One agent, one repo, one subprocess — with the task store pointed at THIS clone."""
    cmd = [sys.executable, "-m", _AGENT_MODULES[agent], "--repo", str(clone), "--project", project]
    if dry_run:
        cmd.append("--dry-run")
    if auto_merge:
        cmd.append("--auto-merge")

    env = os.environ.copy()
    if backend:
        env["TASK_STORE_BACKEND"] = backend
        if backend == "git-bug":
            env["GIT_BUG_TASK_STORE_REPO_PATH"] = str(clone)
            env["GIT_BUG_TASK_STORE_PROJECT"] = project
        elif backend == "github":
            env["GITHUB_TASK_STORE_REPO"] = repo_name

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=str(clone)
        )
    except subprocess.TimeoutExpired:
        return AgentRun(
            repo=repo_name,
            agent=agent,
            status="error",
            detail=f"timed out after {timeout}s",
            exit_code=-1,
        )

    status = _parse_status(proc.stdout, proc.returncode)
    detail = ""
    if status in ("error", "advisory", "conflict") or proc.returncode != 0:
        combined = (proc.stdout + "\n" + proc.stderr).strip()
        detail = combined[-400:]
    return AgentRun(
        repo=repo_name, agent=agent, status=status, detail=detail, exit_code=proc.returncode
    )


def list_github_repos(
    owners: list[str],
    *,
    skip_archived: bool,
    skip_forks: bool,
    timeout: int,
) -> list[str]:
    """``owner/name`` for every listable repo across *owners*, via the gh CLI.

    Archived repos and forks are skipped by default: archives are read-only, and forks
    are what `forge upstream` handles deliberately via SWEEP_UPSTREAM_REMOTES — not what
    a bulk deps sweep should stumble into."""
    names: list[str] = []
    for owner in owners:
        result = subprocess.run(
            ["gh", "repo", "list", owner, "--json", "name,isArchived,isFork", "--limit", "500"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise SweepError(
                f"gh repo list {owner} failed (exit {result.returncode}): {result.stderr.strip()}"
            )
        for repo in json.loads(result.stdout or "[]"):
            if skip_archived and repo.get("isArchived"):
                continue
            if skip_forks and repo.get("isFork"):
                continue
            names.append(f"{owner}/{repo['name']}")
    return names


def github_clone_url(name: str, protocol: str) -> str:
    if protocol == "ssh":
        return f"git@github.com:{name}.git"
    return f"https://github.com/{name}.git"


def prune_workdir(workdir: Path, names: list[str]) -> list[str]:
    """Remove workdir clones whose repo no longer appears in the FULL enumeration.

    Compares against everything the instance listed (not the include/exclude selection —
    a scoped run must never prune repos it merely didn't look at). A clone is any dir with
    a ``.git`` at depth one or two under the workdir."""
    keep = set(names)
    removed: list[str] = []
    if not workdir.is_dir():
        return removed
    for first in sorted(p for p in workdir.iterdir() if p.is_dir()):
        if (first / ".git").exists():
            candidates = [(first.name, first)]
        else:
            candidates = [
                (f"{first.name}/{second.name}", second)
                for second in sorted(p for p in first.iterdir() if p.is_dir())
                if (second / ".git").exists()
            ]
        for name, path in candidates:
            if name not in keep:
                shutil.rmtree(path, ignore_errors=True)
                removed.append(name)
    return removed


def _enumerate(s) -> tuple[list[str], Callable[[str], str], str]:
    """(names, url_for, display_host) for the configured source; raises SweepError."""
    if s.source == "soft-serve":
        if not s.host:
            raise SweepError("SWEEP_HOST is not set")
        names = list_repos(s.host, s.port, timeout=s.ssh_timeout)
        return names, (lambda n: clone_url(s.host, s.port, n)), s.host
    if s.source == "github":
        if not s.github_owners:
            raise SweepError("SWEEP_GITHUB_OWNERS is not set")
        names = list_github_repos(
            s.github_owners,
            skip_archived=s.skip_archived,
            skip_forks=s.skip_forks,
            timeout=s.ssh_timeout,
        )
        display = "github:" + ",".join(s.github_owners)
        return names, (lambda n: github_clone_url(n, s.clone_protocol)), display
    raise SweepError(f"unknown SWEEP_SOURCE {s.source!r} (supported: soft-serve, github)")


def _resolve_backend(s) -> str:
    """The task-store backend injected into agent runs. "auto" follows the source —
    advisories belong where that host's repos keep their tasks."""
    if s.task_store_backend == "auto":
        return "github" if s.source == "github" else "git-bug"
    if s.task_store_backend == "inherit":
        return ""
    return s.task_store_backend


def sweep(
    *,
    dry_run: bool = False,
    auto_merge: bool = False,
    log: Callable[[str], None] = print,
) -> tuple[SweepResult, int]:
    """Run the whole sweep. Returns (result, exit_code): 0 unless a DRIVER-level failure
    (bad source config, enumeration failed, workdir unusable) — per-repo failures are
    rows, not exit codes."""
    s = settings
    try:
        names, url_for, display_host = _enumerate(s)
    except (SweepError, subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
        return SweepResult(host=s.host, errors=[f"enumeration failed: {e}"]), 2

    backend = _resolve_backend(s)
    selected = filter_repos(names, s.include, s.exclude)
    skipped = sorted(set(names) - set(selected))
    log(f"{display_host}: {len(names)} repo(s), {len(selected)} selected")

    try:
        s.workdir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return SweepResult(
            host=display_host,
            repos=selected,
            skipped=skipped,
            errors=[f"workdir unusable: {e}"],
        ), 2

    runs: list[AgentRun] = []
    errors: list[str] = []
    for name in selected:
        dest = s.workdir / name
        try:
            ensure_clone(url_for(name), dest, timeout=s.clone_timeout)
            if backend == "git-bug":
                ensure_bug_identity(dest, name=s.bug_user_name, email=s.bug_user_email)
                warning = bug_sync(dest, "pull", timeout=s.ssh_timeout)
                if warning:
                    log(f"{name}: warning: {warning}")
        except Exception as e:  # noqa: BLE001 — one repo must never stop the sweep
            errors.append(f"{name}: {e}")
            log(f"{name}: SKIPPED — {e}")
            continue

        project = name.rsplit("/", 1)[-1]
        agents: list[str] = []
        if s.deps_enabled:
            # Pre-check the ecosystem so unsupported repos (no uv.lock/go.mod yet) read
            # as benign skips in the summary, not as error rows from the deps agent.
            try:
                detect_ecosystem(dest)
                agents.append("deps")
            except EcosystemError as e:
                runs.append(
                    AgentRun(
                        repo=name,
                        agent="deps",
                        status="skipped",
                        detail=str(e).splitlines()[0][:120],
                    )
                )
                log(f"{name} [deps]: skipped (no supported ecosystem)")
        if s.upstream_enabled and name in s.upstream_remotes:
            try:
                ensure_upstream_remote(dest, s.upstream_remotes[name])
                agents.append("upstream")
            except GitError as e:
                errors.append(f"{name}: upstream remote: {e}")

        for agent in agents:
            try:
                run = run_agent(
                    name,
                    dest,
                    agent,
                    project=project,
                    dry_run=dry_run,
                    auto_merge=auto_merge,
                    backend=backend,
                    timeout=s.run_timeout,
                )
            except Exception as e:  # noqa: BLE001 — same isolation contract
                run = AgentRun(repo=name, agent=agent, status="error", detail=str(e), exit_code=-1)
            runs.append(run)
            log(f"{name} [{agent}]: {run.status}")

        if backend == "git-bug" and not dry_run:
            warning = bug_sync(dest, "push", timeout=s.ssh_timeout)
            if warning:
                log(f"{name}: warning: {warning}")

    if s.prune:
        for removed in prune_workdir(s.workdir, names):
            log(f"pruned stale workdir clone: {removed}")

    result = SweepResult(
        host=display_host, repos=selected, skipped=skipped, runs=runs, errors=errors
    )
    _log(result)
    return result, 0


def _log(result: SweepResult) -> None:
    try:
        log_decision(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "agent": "sweep",
                **result.model_dump(),
            },
            settings.auto_log_path,
        )
    except OSError:
        pass  # a logging failure must never block or crash the sweep
