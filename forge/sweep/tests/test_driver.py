"""Sweep driver: filtering, status parsing, clone lifecycle, env injection, isolation."""

from __future__ import annotations

import subprocess
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from forge.sweep import driver as dr
from forge.sweep.config import settings
from forge.sweep.models import AgentRun

# ---------------------------------------------------------------------------
# filter_repos / _parse_status
# ---------------------------------------------------------------------------


class TestFiltering:
    NAMES = ["me/fork-one", "me/fork-two", "other/thing"]

    def test_include_all_by_default(self):
        assert dr.filter_repos(self.NAMES, ["*"], []) == self.NAMES

    def test_include_glob_selects(self):
        assert dr.filter_repos(self.NAMES, ["me/*"], []) == ["me/fork-one", "me/fork-two"]

    def test_exclude_wins_over_include(self):
        assert dr.filter_repos(self.NAMES, ["*"], ["*two"]) == ["me/fork-one", "other/thing"]


class TestParseStatus:
    def test_deps_headline_parsed(self):
        assert dr._parse_status("stuff\n# meta deps — branched\n- Bump: x", 1) == "branched"

    def test_upstream_headline_parsed(self):
        assert dr._parse_status("# forge upstream — up-to-date\n", 0) == "up-to-date"

    def test_last_headline_wins(self):
        out = "# meta deps — planned\nlater...\n# meta deps — advisory\n"
        assert dr._parse_status(out, 1) == "advisory"

    def test_fallback_by_exit_code(self):
        assert dr._parse_status("no headline", 0) == "ok"
        assert dr._parse_status("traceback...", 3) == "error"


# ---------------------------------------------------------------------------
# ensure_clone — real git against a local bare "server"
# ---------------------------------------------------------------------------


def _g(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True)
    assert result.returncode == 0, f"git {' '.join(args)}: {result.stderr}"
    return result.stdout.strip()


@pytest.fixture
def server(tmp_path):
    """A bare repo with one commit, plus a work clone to push updates from."""
    work = tmp_path / "work"
    work.mkdir()
    _g(work, "init", "-q", "-b", "main")
    _g(work, "config", "user.email", "t@e.c")
    _g(work, "config", "user.name", "T")
    (work / "f.txt").write_text("v1\n")
    _g(work, "add", "-A")
    _g(work, "commit", "-qm", "one")
    bare = tmp_path / "srv.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    _g(work, "remote", "add", "origin", str(bare))
    _g(work, "push", "-q", "-u", "origin", "main")
    return work, bare


class TestEnsureClone:
    def test_clones_on_first_sight(self, server, tmp_path):
        _, bare = server
        dest = tmp_path / "wd" / "me" / "repo"
        dr.ensure_clone(str(bare), dest, timeout=60)
        assert (dest / "f.txt").read_text() == "v1\n"

    def test_refresh_resets_to_origin(self, server, tmp_path):
        work, bare = server
        dest = tmp_path / "wd" / "me" / "repo"
        dr.ensure_clone(str(bare), dest, timeout=60)
        # Server advances; the workdir clone also accumulates local damage.
        (work / "f.txt").write_text("v2\n")
        _g(work, "add", "-A")
        _g(work, "commit", "-qm", "two")
        _g(work, "push", "-q", "origin", "main")
        (dest / "f.txt").write_text("stale local damage\n")
        dr.ensure_clone(str(bare), dest, timeout=60)
        assert (dest / "f.txt").read_text() == "v2\n"  # cache reset, not merged

    def test_clone_failure_raises_giterror(self, tmp_path):
        from forge.shared.gitops import GitError

        with pytest.raises(GitError, match="clone"):
            dr.ensure_clone(str(tmp_path / "nope.git"), tmp_path / "wd" / "x", timeout=60)


class TestEnsureUpstreamRemote:
    def test_adds_then_updates(self, server, tmp_path):
        _, bare = server
        dest = tmp_path / "wd" / "repo"
        dr.ensure_clone(str(bare), dest, timeout=60)
        dr.ensure_upstream_remote(dest, "https://example.com/a.git")
        assert _g(dest, "remote", "get-url", "upstream") == "https://example.com/a.git"
        dr.ensure_upstream_remote(dest, "https://example.com/b.git")
        assert _g(dest, "remote", "get-url", "upstream") == "https://example.com/b.git"


# ---------------------------------------------------------------------------
# run_agent — env injection via a captured subprocess
# ---------------------------------------------------------------------------


class TestRunAgent:
    def test_env_points_task_store_at_the_clone(self, monkeypatch, tmp_path):
        seen = {}

        def fake_run(cmd, **kw):
            seen["cmd"], seen["env"] = cmd, kw["env"]
            return CompletedProcess(cmd, 0, stdout="# meta deps — branched\n", stderr="")

        monkeypatch.setattr(dr.subprocess, "run", fake_run)
        run = dr.run_agent(
            "me/repo",
            tmp_path,
            "deps",
            project="repo",
            dry_run=True,
            auto_merge=False,
            backend="git-bug",
            timeout=60,
        )
        assert run.status == "branched" and run.exit_code == 0
        assert "--dry-run" in seen["cmd"] and "--auto-merge" not in seen["cmd"]
        assert "forge.dependabot.main" in seen["cmd"]
        assert seen["env"]["TASK_STORE_BACKEND"] == "git-bug"
        assert seen["env"]["GIT_BUG_TASK_STORE_REPO_PATH"] == str(tmp_path)
        assert seen["env"]["GIT_BUG_TASK_STORE_PROJECT"] == "repo"

    def test_empty_backend_inherits_caller_env(self, monkeypatch, tmp_path):
        seen = {}

        def fake_run(cmd, **kw):
            seen["env"] = kw["env"]
            return CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(dr.subprocess, "run", fake_run)
        monkeypatch.delenv("TASK_STORE_BACKEND", raising=False)
        dr.run_agent(
            "me/repo",
            tmp_path,
            "upstream",
            project="repo",
            dry_run=False,
            auto_merge=False,
            backend="",
            timeout=60,
        )
        assert "TASK_STORE_BACKEND" not in seen["env"]

    def test_timeout_is_an_error_row_not_a_crash(self, monkeypatch, tmp_path):
        def fake_run(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd, 60)

        monkeypatch.setattr(dr.subprocess, "run", fake_run)
        run = dr.run_agent(
            "me/repo",
            tmp_path,
            "deps",
            project="repo",
            dry_run=False,
            auto_merge=False,
            backend="",
            timeout=60,
        )
        assert run.status == "error" and "timed out" in run.detail

    def test_failure_detail_carries_output_tail(self, monkeypatch, tmp_path):
        def fake_run(cmd, **kw):
            return CompletedProcess(cmd, 1, stdout="", stderr="boom: no repo")

        monkeypatch.setattr(dr.subprocess, "run", fake_run)
        run = dr.run_agent(
            "me/repo",
            tmp_path,
            "deps",
            project="repo",
            dry_run=False,
            auto_merge=False,
            backend="",
            timeout=60,
        )
        assert run.status == "error" and "boom: no repo" in run.detail


# ---------------------------------------------------------------------------
# sweep — orchestration with fakes: selection, isolation, upstream wiring, exit codes
# ---------------------------------------------------------------------------


@pytest.fixture
def swept(monkeypatch, tmp_path):
    """Configure settings + fake collaborators; returns the capture dict."""
    monkeypatch.setattr(settings, "host", "test-host")
    monkeypatch.setattr(settings, "workdir", tmp_path / "wd")
    monkeypatch.setattr(settings, "include", ["*"])
    monkeypatch.setattr(settings, "exclude", [])
    monkeypatch.setattr(settings, "upstream_remotes", {})
    monkeypatch.setattr(settings, "task_store_backend", "")
    monkeypatch.setattr(settings, "auto_log_path", tmp_path / "sweep.jsonl")

    calls = {"clones": [], "agents": [], "remotes": []}
    monkeypatch.setattr(dr, "list_repos", lambda host, port, timeout: ["a/one", "a/two", "a/three"])
    monkeypatch.setattr(
        dr, "ensure_clone", lambda url, dest, timeout: calls["clones"].append(str(dest)) or dest
    )
    monkeypatch.setattr(
        dr, "ensure_upstream_remote", lambda dest, url: calls["remotes"].append((str(dest), url))
    )

    def fake_agent(repo, clone, agent, **kw):
        calls["agents"].append((repo, agent))
        return AgentRun(repo=repo, agent=agent, status="branched")

    monkeypatch.setattr(dr, "run_agent", fake_agent)
    return calls


def test_sweep_runs_deps_on_every_selected_repo(swept):
    result, code = dr.sweep(log=lambda m: None)
    assert code == 0
    assert [(r.repo, r.agent) for r in result.runs] == [
        ("a/one", "deps"),
        ("a/two", "deps"),
        ("a/three", "deps"),
    ]


def test_sweep_fail_isolation_skips_only_the_broken_repo(swept, monkeypatch):
    def clone(url, dest, timeout):
        if "a/two" in str(dest):
            raise RuntimeError("clone exploded")
        return dest

    monkeypatch.setattr(dr, "ensure_clone", clone)
    result, code = dr.sweep(log=lambda m: None)
    assert code == 0  # repo failures are rows, not exit codes
    assert {r.repo for r in result.runs} == {"a/one", "a/three"}
    assert any("a/two" in e and "clone exploded" in e for e in result.errors)


def test_sweep_wires_upstream_only_for_configured_forks(swept, monkeypatch):
    monkeypatch.setattr(settings, "upstream_remotes", {"a/one": "https://example.com/up.git"})
    result, _ = dr.sweep(log=lambda m: None)
    assert ("a/one", "upstream") in swept["agents"]
    assert ("a/two", "upstream") not in swept["agents"]
    assert swept["remotes"] and swept["remotes"][0][1] == "https://example.com/up.git"


def test_sweep_include_exclude(swept, monkeypatch):
    monkeypatch.setattr(settings, "exclude", ["*three"])
    result, _ = dr.sweep(log=lambda m: None)
    assert {r.repo for r in result.runs} == {"a/one", "a/two"}
    assert result.skipped == ["a/three"]


def test_missing_host_is_a_driver_failure(swept, monkeypatch):
    monkeypatch.setattr(settings, "host", "")
    result, code = dr.sweep(log=lambda m: None)
    assert code == 2 and "SWEEP_HOST" in result.errors[0]


def test_enumeration_failure_is_a_driver_failure(swept, monkeypatch):
    def boom(host, port, timeout):
        raise dr.SweepError("ssh unreachable")

    monkeypatch.setattr(dr, "list_repos", boom)
    result, code = dr.sweep(log=lambda m: None)
    assert code == 2 and "ssh unreachable" in result.errors[0]


def test_sweep_writes_the_decision_log(swept, tmp_path):
    dr.sweep(log=lambda m: None)
    assert (tmp_path / "sweep.jsonl").exists()
