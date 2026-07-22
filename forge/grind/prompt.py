"""Build the per-turn instruction spec the model reads.

Each turn the model gets: the repo's durable lessons, the goal, the scope it may edit, the loop's
rules (make one focused change; the harness — not you — runs the cycle; never commit), and the
latest run's observation. Kept plain so it survives being written to a file and read back.
"""

from __future__ import annotations

from pathlib import Path

from forge.grind.models import GrindConfig
from forge.shared.lessons import read_lessons, render_lessons_preamble

_RULES = """## Rules

- Make ONE focused change to move toward the goal — the smallest edit that could plausibly help.
- Edit ONLY source files{scope}. Do not edit test fixtures or unrelated files.
- Do NOT run the experiment yourself — the harness runs reset, load, the migration, and the check
  for you, and shows you the result below. Just change the code.
- Do NOT commit, and do NOT touch version control.
- If you genuinely cannot proceed (missing info, the goal is unreachable as stated), print a line
  starting with BLOCKED: explaining why, and stop.
"""


def build_spec(cfg: GrindConfig, repo: Path, observation: str, iteration: int) -> str:
    """Assemble the instruction spec for one edit turn."""
    preamble = render_lessons_preamble(read_lessons(repo))
    scope = ""
    if cfg.edit_paths:
        scope = " under: " + ", ".join(cfg.edit_paths)
    rules = _RULES.format(scope=scope)
    obs = observation.strip() or "(no output captured)"
    return (
        f"{preamble}# Grind — iterate toward a goal (turn {iteration})\n\n"
        f"## Goal\n\n{cfg.goal.strip()}\n\n"
        f"{rules}\n"
        f"## Latest run\n\n"
        f"This is what the experiment produced on the current code. Diagnose it and make your "
        f"one change.\n\n{obs}\n"
    )
