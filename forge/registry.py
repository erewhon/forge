"""The agent registry — the single source of truth for the `forge` front door.

Each :class:`AgentCommand` maps a verb (e.g. ``research``) to an agent package whose ``main(argv)``
runs it. Both the CLI dispatcher (``forge.cli``) and the MCP server (``forge.mcp_server``) iterate
this list, so adding an agent is one entry that lights it up everywhere it opts into. ``main`` is
loaded lazily so ``forge --help`` never imports an agent's SDK/config until its verb actually runs.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass

# An agent entrypoint: takes an argv list (or None → sys.argv) and returns a process exit code.
AgentMain = Callable[[list[str] | None], int | None]


@dataclass(frozen=True)
class AgentCommand:
    name: str  # CLI verb / MCP tool stem, e.g. "research"
    summary: str  # one-line help shown by `forge --help`
    module: str  # dotted path to the agent main module, e.g. "forge.general_researcher.main"
    exposes_cli: bool = True
    exposes_mcp: bool = True

    def load_main(self) -> AgentMain:
        """Import the agent module on demand and return its ``main`` callable."""
        return importlib.import_module(self.module).main


REGISTRY: list[AgentCommand] = [
    AgentCommand(
        name="research",
        summary="Iterative research harness (plan → research → verify → synthesize).",
        module="forge.general_researcher.main",
    ),
    AgentCommand(
        name="book",
        summary="Book research harness (generator–evaluator sprint cycles).",
        module="forge.book_researcher.main",
    ),
    AgentCommand(
        name="edit",
        summary="Run one prompt against a jj repo with N models, then compare.",
        module="forge.parallel_edit.main",
        # Not exposed over MCP: it mutates a real jj repo and spawns model subprocesses whose
        # stdout would escape the server's capture and corrupt the MCP stdio protocol. CLI-only.
        exposes_mcp=False,
    ),
    AgentCommand(
        name="review",
        summary="PR review ensemble (review / digest / supply-chain lenses).",
        module="forge.pr_review_ensemble.main",
    ),
    # The remaining on-demand agents are CLI-only (exposes_mcp=False). `task` autonomously edits +
    # commits repos (like `edit`); the others write to Forge tasks / files and are human-run, not
    # the sort of thing to call mid-task. Typed MCP wrappers can be added later if any prove useful.
    AgentCommand(
        name="audit",
        summary="Adversarial multi-model code audit (discover → dedup → verify) of files/dirs.",
        module="forge.code_audit_ensemble.main",
        # Read-only LLM analysis; an MCP wrapper is plausible later, but CLI-only for the MVP.
        exposes_mcp=False,
    ),
    AgentCommand(
        name="testing",
        summary="Multi-model test-coverage review: find untested behavior, suggest tests.",
        module="forge.testing_ensemble.main",
        exposes_mcp=False,
    ),
    AgentCommand(
        name="refactor",
        summary="Multi-model refactoring review: find smells, verify safe+worthwhile, plan them.",
        module="forge.refactor_ensemble.main",
        exposes_mcp=False,
    ),
    AgentCommand(
        name="code-review",
        summary="Nightly code review of recent commits.",
        module="forge.code_reviewer.main",
        exposes_mcp=False,
    ),
    AgentCommand(
        name="task",
        summary="Autonomous Forge task worker (picks, executes, commits).",
        module="forge.task_worker.main",
        exposes_mcp=False,
    ),
    AgentCommand(
        name="grind",
        summary="Iterate a goal via a runbook loop (reset -> run -> check -> adjust); no commits.",
        module="forge.grind.main",
        # CLI-only like task/edit: mutates a working copy and spawns model subprocesses.
        exposes_mcp=False,
    ),
    AgentCommand(
        name="queue",
        summary="Backlog report: non-done tasks per project, worker-readiness resolved.",
        module="forge.queue.main",
        # Read-only, but agents already have richer task queries via the Nous MCP; CLI-only.
        exposes_mcp=False,
    ),
    AgentCommand(
        name="build",
        summary="Coding pipeline: plan, run, and inspect epic builds.",
        module="forge.coding_pipeline.main",
        exposes_mcp=False,
    ),
    AgentCommand(
        name="evals",
        summary="Judgment eval harness: score models against frozen gold sets.",
        module="forge.evals.main",
        exposes_mcp=False,
    ),
    AgentCommand(
        name="deps",
        summary="Dependency bumper: scan, gate, and auto-merge clean low-risk bumps.",
        module="forge.dependabot.main",
        exposes_mcp=False,
    ),
    AgentCommand(
        name="upstream",
        summary="Upstream sync for additive forks: fetch, merge, gate, push.",
        module="forge.upstream_sync.main",
        # CLI-only like deps: it pushes branches to real remotes.
        exposes_mcp=False,
    ),
    AgentCommand(
        name="sweep",
        summary="Fleet sweep: run deps/upstream across every repo on a Soft Serve instance.",
        module="forge.sweep.main",
        # CLI-only: it clones repos and spawns agents that push to real remotes.
        exposes_mcp=False,
    ),
]
