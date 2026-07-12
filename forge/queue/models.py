"""The queue report's row model — one non-done task with its dispatch state resolved.

``QueueRow`` is what ``TaskStore.queue()`` returns: a cross-project cousin of the wave
planner's ``LeafRow`` that additionally carries the project (the report groups by it),
the feature (so a row reads as "which epic"), and the model tier (so an auto row shows
what would execute it). Dispatchability is the worker's exact gate — status Ready AND
an auto execution mode AND unblocked — so the report never claims the worker would pick
up a task it wouldn't.
"""

from __future__ import annotations

from pydantic import BaseModel

from forge.shared.task_conventions import AUTO_MODES


class QueueRow(BaseModel):
    """A normalized non-done task row (null-as-manual applied, blocked state resolved)."""

    project: str
    task: str
    status: str
    execution_mode: str = "Manual"
    priority: int = 99
    blocked: bool = False
    blocked_by: list[str] = []
    feature: str = ""
    model_tier: str = ""

    @property
    def is_auto(self) -> bool:
        """Tagged for autonomous execution (Auto-OK / Auto-Preferred)."""
        return self.execution_mode.lower() in AUTO_MODES

    @property
    def is_dispatchable(self) -> bool:
        """Would the task worker pick this up right now? (Ready AND auto AND unblocked.)"""
        return self.status == "Ready" and self.is_auto and not self.blocked
