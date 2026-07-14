# forge — agent notes

Front door for the erewhon code agents: the `forge` CLI and `forge-mcp` MCP server over the
agent packages in `forge/<package>/`. Human-facing docs live in `README.md`; per-package
READMEs cover each agent.

## Conventions

- Python 3.12+, uv-managed. Run everything through `uv run ...` from the repo root.
- Tests: `uv run pytest forge/<package>` for one package, `uv run pytest` for the suite.
- Lint/format: `uv run ruff check .` and `uv run ruff format .` (line length 100).
- Typed code with pydantic models throughout. Settings are pydantic-settings classes with
  per-agent env prefixes; a `.env` at the repo root holds machine-local values and is never
  committed.
- Nous integration is optional (the `[nous]` extra). Anything touching it must import
  lazily/guarded — see `forge/shared/task_store.py` and `forge/task_worker/nous_client.py`
  for the pattern. `TASK_STORE_BACKEND=github` must always work without nous installed.
- Pure signal/parse helpers stay separate from network fetchers, and fetchers are injectable
  for tests (see `forge/dependabot/supply_chain.py`).
- Write errors for the next attempt, not the stack-trace reader: any error on a path a loop
  retries (worker outcomes, task-store writes, dispatch preflight, executor failures) must state
  the failed expectation and the likely remedy, then the evidence — never a bare status or
  exception. See `docs/error-surface-audit.md`.

## Boundaries for autonomous workers

- Do not modify `.task_worker/` or anything inside it (worker plumbing and task specs).
- Do not commit `.env` or log files.
- Keep the diff scoped to the files named in the task spec.
