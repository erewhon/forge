"""Tests for the meta-agents MCP server — tools build argv and capture output (no agents run)."""

from __future__ import annotations

import json
from pathlib import Path

from agents import mcp_server


def test_list_agents_reports_registry_with_edit_cli_only():
    agents = json.loads(mcp_server.list_agents())
    names = {a["name"] for a in agents}
    assert {"research", "book", "edit", "review"} <= names
    edit = next(a for a in agents if a["name"] == "edit")
    assert edit["cli"] is True and edit["mcp"] is False


def test_research_builds_argv(monkeypatch):
    seen: dict = {}

    def fake(main, argv):
        seen["argv"] = argv
        return 0, "planned"

    monkeypatch.setattr(mcp_server, "_run_captured", fake)
    out = json.loads(mcp_server.research("Why X?", max_sprints=2, dry_run=True))
    assert seen["argv"] == ["Why X?", "--max-sprints", "2", "--dry-run"]
    assert out["exit_code"] == 0
    assert out["answer"] is None  # no synthesis.md written by the fake


def test_review_builds_argv_and_reads_advisory(monkeypatch):
    seen: dict = {}

    def fake(main, argv):
        seen["argv"] = argv
        out_path = argv[argv.index("--output") + 1]
        Path(out_path).write_text("# Advisory\nLGTM")
        return 0, "ran"

    monkeypatch.setattr(mcp_server, "_run_captured", fake)
    out = json.loads(mcp_server.review("a diff", lens="digest", pr_ref="PR-1"))
    argv = seen["argv"]
    assert argv[argv.index("--pass") + 1] == "digest"
    assert argv[argv.index("--pr-ref") + 1] == "PR-1"
    assert out["advisory"] == "# Advisory\nLGTM"


def test_book_builds_argv(monkeypatch):
    seen: dict = {}

    def fake(main, argv):
        seen["argv"] = argv
        return 0, "ok"

    monkeypatch.setattr(mcp_server, "_run_captured", fake)
    mcp_server.book("/path/book.yaml", max_sprints=1)
    assert seen["argv"] == ["/path/book.yaml", "--max-sprints", "1"]


def test_run_captured_none_return_is_zero():
    code, _ = mcp_server._run_captured(lambda argv: None, [])
    assert code == 0


def test_run_captured_captures_stdout_and_does_not_leak():
    def noisy(argv):
        print("progress line to stdout")
        return 0

    code, log = mcp_server._run_captured(noisy, [])
    assert code == 0
    assert "progress line to stdout" in log  # captured, not leaked to real stdout


def test_run_captured_swallows_exception():
    def boom(argv):
        raise RuntimeError("router down")

    code, log = mcp_server._run_captured(boom, [])
    assert code == 1
    assert "router down" in log


def test_run_captured_propagates_systemexit_code():
    def bail(argv):
        raise SystemExit(2)  # what argparse does on a bad arg

    code, _ = mcp_server._run_captured(bail, [])
    assert code == 2
