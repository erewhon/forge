"""`forge` MCP server — the erewhon code agents as callable tools.

Mirrors the nous MCP server: FastMCP over stdio, one ``@mcp.tool()`` per exposed agent. Each tool
builds an argv list and calls the agent's ``main(argv)`` in-process with stdout/stderr **captured**
— so the agent's progress prints never corrupt the MCP stdio JSON-RPC stream — then returns the
captured log plus the agent's real deliverable (synthesized answer, advisory markdown, ...).

The registry (``forge.registry``) is the shared spine: ``list_agents`` reports it, and only agents
with ``exposes_mcp=True`` get a tool here (``edit`` is CLI-only — it mutates a repo and spawns
subprocesses that would escape capture).
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from forge.registry import REGISTRY, AgentMain

mcp = FastMCP(
    "forge",
    instructions=(
        "Callable wrappers around the erewhon code agents. `research` runs the iterative research "
        "harness on a question and returns a synthesized answer; `review` runs the PR review "
        "ensemble on a unified diff; `book` runs the book researcher. These are synchronous and "
        "can be slow (multiple LLM calls). Call `list_agents` to see what the front door exposes."
    ),
)


def _run_captured(main: AgentMain, argv: list[str]) -> tuple[int, str]:
    """Call an agent ``main(argv)`` with stdout+stderr captured; return (exit_code, captured_text).

    Capture is mandatory here: a stray ``print`` to real stdout would corrupt the MCP stdio
    protocol. Any exception (or argparse ``SystemExit``) is folded into the log and a non-zero code,
    so an agent failure never tears down the server.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            code = main(argv) or 0
        except SystemExit as e:  # argparse errors raise SystemExit(2)
            code = e.code if isinstance(e.code, int) else 1
        except Exception as e:  # noqa: BLE001 — keep the server alive on any agent error
            buf.write(f"\n[error] {type(e).__name__}: {e}\n")
            code = 1
    return code, buf.getvalue()


@mcp.tool()
def list_agents() -> str:
    """List the agents the forge front door exposes (name, summary, and the surfaces they opt into).

    Returns a JSON array of {name, summary, cli, mcp}.
    """
    return json.dumps(
        [
            {"name": c.name, "summary": c.summary, "cli": c.exposes_cli, "mcp": c.exposes_mcp}
            for c in REGISTRY
        ],
        indent=2,
    )


@mcp.tool()
def research(question: str, max_sprints: int | None = None, dry_run: bool = False) -> str:
    """Run the iterative research agent (plan → research → verify → synthesize) on a question.

    Synchronous and potentially long (multiple LLM sprints — cap with `max_sprints`). Set
    `dry_run=True` to plan sprints without any LLM calls. Returns JSON with the synthesized markdown
    `answer` (null if none was produced) and the tail of the run `log`.
    """
    from forge.general_researcher.main import _load_topic_config, _topic_dir
    from forge.general_researcher.main import main as research_main

    argv = [question]
    if max_sprints is not None:
        argv += ["--max-sprints", str(max_sprints)]
    if dry_run:
        argv.append("--dry-run")

    code, log = _run_captured(research_main, argv)
    synthesis = _topic_dir(_load_topic_config(question)) / "synthesis.md"
    answer = synthesis.read_text() if synthesis.is_file() else None
    return json.dumps({"exit_code": code, "answer": answer, "log": log[-4000:]})


@mcp.tool()
def review(diff: str, lens: str = "review", pr_ref: str = "adhoc") -> str:
    """Run the PR review ensemble on a unified diff.

    `lens` is one of 'review' (advisory), 'digest' (navigational summary of a large PR), or
    'supply-chain' (dependency/hook/CI/obfuscation audit). Returns JSON with the `advisory` markdown
    (null if none produced) and the tail of the run `log`.
    """
    from forge.pr_review_ensemble.main import main as review_main

    with tempfile.TemporaryDirectory() as td:
        diff_path = Path(td) / "pr.diff"
        out_path = Path(td) / "advisory.md"
        diff_path.write_text(diff)
        argv = [
            "--pass",
            lens,
            "--pr-ref",
            pr_ref,
            "--diff-file",
            str(diff_path),
            "--output",
            str(out_path),
        ]
        code, log = _run_captured(review_main, argv)
        advisory = out_path.read_text() if out_path.is_file() and out_path.stat().st_size else None
    return json.dumps({"exit_code": code, "advisory": advisory, "log": log[-2000:]})


@mcp.tool()
def book(config_path: str, max_sprints: int | None = None, dry_run: bool = False) -> str:
    """Run the book research agent on a book config (path to a YAML/JSON file).

    Set `dry_run=True` to plan sprints without LLM calls. Returns JSON with the run `log` tail.
    """
    from forge.book_researcher.main import main as book_main

    argv = [config_path]
    if max_sprints is not None:
        argv += ["--max-sprints", str(max_sprints)]
    if dry_run:
        argv.append("--dry-run")

    code, log = _run_captured(book_main, argv)
    return json.dumps({"exit_code": code, "log": log[-4000:]})


def main() -> None:
    """Console-script entry point (``[project.scripts] forge-mcp``)."""
    # Layer ~/.config/forge/config.toml into the environment before any agent tool is invoked
    # (agents and their settings are imported lazily inside the tool bodies).
    from forge.shared.user_config import apply_user_config

    apply_user_config()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
