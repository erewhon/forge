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
]
