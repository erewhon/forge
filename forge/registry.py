"""The agent registry — the single source of truth for the `meta` front door.

Each :class:`AgentCommand` maps a verb (e.g. ``research``) to an agent package whose ``main(argv)``
runs it. Both the CLI dispatcher (``agents.cli``) and, later, the MCP server iterate this list, so
adding an agent is one entry that lights it up everywhere it opts into. ``main`` is loaded lazily so
``meta --help`` never imports an agent's SDK/config until its verb actually runs.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass

# An agent entrypoint: takes an argv list (or None → sys.argv) and returns a process exit code.
AgentMain = Callable[[list[str] | None], int | None]


@dataclass(frozen=True)
class AgentCommand:
    name: str  # CLI verb / future MCP tool stem, e.g. "research"
    summary: str  # one-line help shown by `meta --help`
    module: str  # dotted path to the agent main module, e.g. "agents.general_researcher.main"
    exposes_cli: bool = True
    exposes_mcp: bool = True

    def load_main(self) -> AgentMain:
        """Import the agent module on demand and return its ``main`` callable."""
        return importlib.import_module(self.module).main


# Phase 1: the top interactive verbs. Later phases add the remaining on-demand agents and the
# scheduled daemons' manual-run verbs.
REGISTRY: list[AgentCommand] = [
    AgentCommand(
        name="research",
        summary="Iterative research harness (plan → research → verify → synthesize).",
        module="agents.general_researcher.main",
    ),
    AgentCommand(
        name="book",
        summary="Book research harness (generator–evaluator sprint cycles).",
        module="agents.book_researcher.main",
    ),
    AgentCommand(
        name="edit",
        summary="Run one prompt against a jj repo with N models, then compare.",
        module="agents.parallel_edit.main",
        # Not exposed over MCP: it mutates a real jj repo and spawns model subprocesses whose
        # stdout would escape the server's capture and corrupt the MCP stdio protocol. CLI-only.
        exposes_mcp=False,
    ),
    AgentCommand(
        name="review",
        summary="PR review ensemble (review / digest / supply-chain lenses).",
        module="agents.pr_review_ensemble.main",
    ),
    # The remaining on-demand agents are CLI-only for now (exposes_mcp=False). `task` autonomously
    # edits + commits repos (like `edit`); the others write to Nous / files and are human-run, not
    # the sort of thing to call mid-task. Typed MCP wrappers can be added later if any prove useful.
    AgentCommand(
        name="audit",
        summary="Adversarial multi-model code audit (discover → dedup → verify) of files/dirs.",
        module="agents.code_audit_ensemble.main",
        # Read-only LLM analysis; an MCP wrapper is plausible later, but CLI-only for the MVP.
        exposes_mcp=False,
    ),
    AgentCommand(
        name="testing",
        summary="Multi-model test-coverage review: find untested behavior, suggest tests.",
        module="agents.testing_ensemble.main",
        exposes_mcp=False,
    ),
    AgentCommand(
        name="refactor",
        summary="Multi-model refactoring review: find smells, verify safe+worthwhile, plan them.",
        module="agents.refactor_ensemble.main",
        exposes_mcp=False,
    ),
    AgentCommand(
        name="code-review",
        summary="Nightly code review of recent commits.",
        module="agents.code_reviewer.main",
        exposes_mcp=False,
    ),
    AgentCommand(
        name="task",
        summary="Autonomous Forge task worker (picks, executes, commits).",
        module="agents.task_worker.main",
        exposes_mcp=False,
    ),
    AgentCommand(
        name="build",
        summary="Coding pipeline: plan, run, and inspect epic builds.",
        module="agents.coding_pipeline.main",
        exposes_mcp=False,
    ),
    AgentCommand(
        name="astro",
        summary="Scout astronomy events worth streaming; write prep pages to Nous.",
        module="agents.astro_scout.main",
        exposes_mcp=False,
    ),
    AgentCommand(
        name="stream",
        summary="Plan weekly astronomy streams from weather + events.",
        module="agents.stream_planner.main",
        exposes_mcp=False,
    ),
    AgentCommand(
        name="dashboard",
        summary="Generate the weekly life dashboard page.",
        module="agents.dashboard.main",
        exposes_mcp=False,
    ),
    AgentCommand(
        name="evals",
        summary="Judgment eval harness: score models against frozen gold sets.",
        module="agents.evals.main",
        exposes_mcp=False,
    ),
]
