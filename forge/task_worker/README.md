# Task Worker

Autonomous execution of Nous tasks explicitly marked for automation. Runs tasks
via OpenCode + local LLM inside a sandboxed `gaol dx` container, commits the
result on the host, and updates the task status — bailing safely if anything
goes sideways.

## Architecture

**Execution is sandboxed; VCS is host-only.**

- The **dx container** (nspawn) has the repo bind-mounted, plus OpenCode,
  uv/pnpm/cargo, and the LLM router DNS entry. Everything OpenCode does
  happens inside this container — it cannot reach other projects, your home
  directory, or anything else on the host.
- The **host** runs all VCS operations (clean-WC check, diff inspection,
  commit, revert). That way the container can't rewrite jj/git history.
- The **task spec** is written to `<repo>/.task_worker/spec-<id>-<uuid>.md`
  (gitignored) inside the bind-mounted repo. OpenCode reads that file instead
  of having multi-KB markdown passed through shell quoting.

## Pipeline

1. **Find task** — scans Nous for `worker_ready=True`, sorts Auto-Preferred
   first then by priority ascending.
2. **Fetch spec** — full task metadata + page content from Nous.
3. **VCS detection** — host checks for `.jj/` or `.git/`.
4. **dx preflight** — `gaol dx info` must report `Status: running`. If not,
   the task is left Ready and the worker exits cleanly.
5. **Clean-WC guard** — bails if the project has uncommitted changes (so the
   worker never collides with your in-progress work).
6. **Mark In Progress** — updates Nous task status.
7. **Execute** — `gaol dx run -- opencode run -m llm/<tier>
   --dangerously-skip-permissions "Read .task_worker/spec-<id>.md and execute it"`.
8. **Scope guard** — host inspects diff, bails and reverts if changes exceed
   `max_files`.
9. **Static checks** — ALWAYS run, independent of `requires_tests`. Additive
   per detected language: `go build ./...` (go.mod), `cargo build`
   (Cargo.toml), `pnpm exec tsc --noEmit` (tsconfig.json), `shellcheck` over
   changed `*.sh`, and a `py_compile` syntax floor over changed `*.py`.
   Commands run in the sandbox, so a detected language whose tool is missing
   there fails closed naming the tool — never a silent pass. (The lint gate,
   `linter.py`, is ruff/Python-only; other languages rely on this gate plus
   their test runner.)
10. **Test** — auto-detects `Justfile` / go.mod (`go build` + `go test`) /
    `pyproject.toml` (pytest) / `package.json` scripts.test / `Cargo.toml`
    and runs via `gaol dx run --` inside the container if
    `requires_tests=Yes`. A Python repo without pytest gets the syntax floor
    instead of a pass; a repo with no runner at all passes with a reason
    string naming every probe — a disclosed decision, never a silent one.
11. **Commit** — on host. `jj describe/new` or `git add/commit` with `auto:
    <title>` message (prefix configurable).
12. **Mark Done** — writes diagnostic notes back to Nous including commit id
    and duration.

**Any failure reverts first (on host), then marks the task back to Ready with
a diagnostic note.** The worker never leaves the repo in a broken state.

## Requirements per project

Before the worker can touch a project, it needs:

1. A VCS (jj or git) — already the case for most active projects.
2. A running `gaol dx` container — set up once via `cd <project> && gaol dx
   shell`. The container must have:
   - `opencode` (usually inherited from the dx base)
   - `jj` (if the project uses jj) — install via cargo or apt inside the
     container
3. At least one task in Nous with `Execution Mode = Auto-OK` or
   `Auto-Preferred`.

## Task metadata the worker reads

The 7 autonomy fields on each Nous task:

| Field | Purpose | Default |
|---|---|---|
| `Execution Mode` | Gate — only `Auto-OK` / `Auto-Preferred` get picked up | `Manual` (null-as-manual) |
| `Model Tier` | Router alias: `auto` / `auto-free` / `auto-full` | `auto` |
| `Estimate` | `xs` / `s` / `m` / `l` / `xl` — not yet enforced, used by planner | null |
| `Complexity` | `routine` / `novel` — hint for escalation | null |
| `Task Type` | `bug-fix` / `feature` / `refactor` / `docs` / `test` / `chore` | null |
| `Max Files` | Soft scope guardrail (bails if diff exceeds) | `default_max_files` (5) |
| `Requires Tests` | Must tests pass before marking Done? | Yes (null-as-true for safety) |

## Usage

```bash
# Dry run — executes OpenCode but won't commit or update Nous
forge/task_worker/run.sh --dry-run

# Pick from any project with a worker-ready task
forge/task_worker/run.sh

# Restrict to one project
forge/task_worker/run.sh --project example
```

## Configuration

Environment variables (all prefixed `TASK_WORKER_`; a gitignored `.env` at the repo
root is also read when running from the repo):

- `PROJECTS_DIR` — default `~/projects`
- `NOTEBOOK_NAME` — default `Forge`
- `DATABASE_NAME` — default `Project Tasks`
- `DAEMON_URL` — default `http://127.0.0.1:7667`
- `TASK_TIMEOUT_SECONDS` — per-task cap, default `1800` (30 min)
- `DEFAULT_MAX_FILES` — fallback when task has no `Max Files` set, default `5`
- `MODEL_TIER_DEFAULT` — default `auto`
- `ALLOWED_PROJECTS` — comma-separated allowlist (empty = no restriction)
- `COMMIT_PREFIX` — default `auto: `
- `DRY_RUN` — default `false`

## Files

- `config.py` — settings
- `models.py` — `TaskInfo`, `ExecutionResult`
- `nous_client.py` — reads tasks via `NousStorage` direct-import; writes via
  `NousDaemonClient` HTTP
- `executor.py` — shells out to `opencode run`
- `vcs.py` — jj / git detect, diff, commit, revert
- `tester.py` — project-agnostic test runner
- `main.py` — orchestrator
- `run.sh` — uv wrapper

## Safety notes

- `--dangerously-skip-permissions` is used in OpenCode. The `max_files` guard
  is the safety net — start with small numbers (2-3) for early tasks.
- Working copy must be clean before the worker runs. It will not stash or
  discard your in-progress changes.
- `git revert` uses `git clean -fd` which is capped at 20 files for extra
  safety.
- For jj repos, `jj restore` reverts the working copy to the parent change.

## Planned next steps

- `/loop` wrapper to run periodically during the day
- Systemd timer for unattended overnight runs
- Model tier escalation — retry with `auto-full` on local-LLM failure
- Morning planner that reports what the worker did overnight vs what's left
