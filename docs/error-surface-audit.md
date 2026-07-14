# Error-surface audit — writing errors for the next attempt

**Principle (loop-engineering):** in an unattended loop, an error message is *fuel for the next
attempt*. `missing repo scope — request the repo scope` lets the next beat self-fix; `Error 403`
wastes attempts. Every error on a path a loop **retries** should name the **failed expectation**
and the **likely remedy**, then the evidence — not a bare status, exception, or output tail.

This audit swept the error surfaces an autonomous coding loop actually consumes and classifies each
as **self-fixable** (says what to do), **diagnosable** (names the expectation, not the remedy), or
**opaque** (bare status/exception — the retry learns nothing).

## What was fixed

The opaque messages on retry paths — where a blind retry consumes the string — were rewritten to
name the expectation + remedy:

| Surface | Before | After (shape) |
|---|---|---|
| `task_worker/main.py` dx-not-ready | reason: `dx container not ready (...)`; remedy only in a `print` | remedy now **in the reason**: `… — provision it with \`gaol dx shell\`, then re-dispatch` |
| `task_worker/main.py` opencode/static/lint/tests gate failures | `<gate> failed: <raw tail>` | `<gate> gate failed — <expectation + fix>. Evidence:\n<tail>` (via `_gate_failure`) |
| `task_worker/main.py` spec-fetch / VCS-inspection / commit failures | bare `: {e}` | names the expectation (unreachable store, dirty/locked repo) + `Detail: {e}` |
| `coding_pipeline/dispatch.py` `worker crashed` / `workspace setup failed` | bare `: {e}` | names it as an infra/sandbox fault + likely cause + `Detail: {e}` |
| `task_worker/dx.py`, `sandbox.py` gaol probes | `gaol dx info failed: {e}` | `… — the gaol daemon is unreachable or not installed; ensure gaol is running and on PATH` |
| `shared/ensemble/executor.py` generic failure | `{type}: {exc}` | appends the retry disposition: `[terminal — will fail over, not retry]` / `[transient — will retry]` |

## Already good (kept)

Config/wiring errors were largely self-fixable already — they list valid values or the fix:
`unknown TASK_STORE_BACKEND {x} (known: 'forge', 'github', 'git-bug')`, the `forge[nous]` install
hint, `gate sub-epics separately or raise CODING_PIPELINE_EPIC_GATE_MAX_CHUNKS`, `Run \`meta build
plan\` first`. Diagnosable store lookups (`task {title!r} not found in the issue tracker`) name the
expectation and forward the backend's own stderr, which usually carries the remedy.

## Convention

Added to `AGENTS.md`:

> **Write errors for the next attempt, not the stack-trace reader.** Any error on a path a loop
> retries must state the failed expectation and the likely remedy, then the evidence — never a bare
> status or exception.

Sibling to the idempotent-writes convention (stable external_refs, idempotent emit, one branch per
bump): idempotent writes are the *state* half of connector design, descriptive errors the *feedback*
half.
