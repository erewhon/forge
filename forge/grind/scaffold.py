"""`forge grind init` — write a ready-to-edit grind runbook that already validates."""

from __future__ import annotations

from pathlib import Path

DEFAULT_FILENAME = "grind.yaml"

_SKELETON = """\
# A grind runbook: a goal + the experiment cycle to iterate on it, with no commits.
# Run:  forge grind ./grind.yaml            (or --dry-run to just print the plan)

goal: >
  Get the data migration to run cleanly against dev data — every table migrated,
  row counts reconciled, no errors.

# The experiment cycle, run in order every turn (shell commands, in the repo root).
steps:
  - name: reset
    run: ./gradlew flywayClean
  - name: load
    run: ./scripts/load-dev-data.sh
  - name: migrate
    run: ./gradlew runMigration

# The machine-checkable done-signal: exit 0 means the goal is met.
# Optional: print a number and capture it with score_regex to unlock hill-climbing
# (grind keeps the best-scoring iteration and rolls back regressions on the jj op log).
check:
  run: ./scripts/verify-migration.sh
  score_regex: "RECONCILED=([0-9]+)"   # e.g. rows reconciled; higher is better
  score_goal: max

# Which step outputs the model sees each turn (default: all steps + the check).
observe: [migrate, check]

# Where the model may edit (a scope hint; optional).
edit_paths: [src/main/kotlin/migration]

# OpenCode model string, passed verbatim to `opencode run -m` (override with --model).
# model: opencode/anthropic/claude-sonnet-4

max_iterations: 20
"""


def write_skeleton(path: str | None, *, force: bool = False) -> Path:
    """Write the skeleton runbook. A directory (or trailing slash) gets ``grind.yaml`` appended."""
    target = Path(path).expanduser() if path else Path(DEFAULT_FILENAME)
    if path and (str(path).endswith("/") or target.is_dir()):
        target = target / DEFAULT_FILENAME
    if target.exists() and not force:
        raise FileExistsError(str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_SKELETON, encoding="utf-8")
    return target
