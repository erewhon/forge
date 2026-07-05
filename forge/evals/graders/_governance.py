"""Apply production's deterministic tag governance before grading leaf shape.

Production never ships raw model leaves: ``_apply_conservative_tags`` (the
post-rule both ``decompose()`` and ``replan()`` run) floors a non-Manual
leaf's ``requires_tests`` to True and ``max_files`` to >= 3, demotes
Spec-Needed/novel leaves to Manual, and floors bare-auto model tiers. Grading
the raw output penalizes the model for fields governance owns — a
false-negative class, not signal. Graders call :func:`floor_to_shipped` on
emitted leaves before any worker-shape / L4 check, so the eval scores what
actually ships. Action kinds, targets, and schema validity are NOT floored —
those stay graded on raw output.
"""

from __future__ import annotations

import json
from pathlib import Path

from agents.coding_pipeline.architect import _apply_conservative_tags
from agents.coding_pipeline.models import FramingProposal, LeafSpec

# _apply_conservative_tags only reads framing.epic_slug (the default-feature
# fallback); when a case has no framing.json the governance floors still apply.
_STUB_FRAMING = FramingProposal(
    goal_as_stated="",
    restated_goal="",
    recommendation="",
    epic_slug="eval-governance-stub",
)


def floor_to_shipped(leaves: list[LeafSpec], case_dir: Path) -> list[LeafSpec]:
    """Mutate *leaves* to their post-governance (shipped) state and return them."""
    framing = _STUB_FRAMING
    framing_path = case_dir / "framing.json"
    if framing_path.is_file():
        try:
            framing = FramingProposal.model_validate(json.loads(framing_path.read_text()))
        except (json.JSONDecodeError, ValueError):
            pass  # unreadable framing -> stub; the floors themselves don't depend on it
    _apply_conservative_tags(leaves, framing)
    return leaves
