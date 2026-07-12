# forge sweep

Fleet sweep: run the per-repo agents across every repo on a Soft Serve instance.

## What it does

1. **Enumerate** — `ssh -p <port> <host> repo list` (Soft Serve's SSH CLI)
2. **Filter** — fnmatch include/exclude globs on repo names
3. **Clone/refresh** — a machine-owned workdir of clones: clone on first sight, `fetch` +
   `reset --hard origin/<default>` on later sweeps (the workdir is a cache; anything of
   value was pushed by the agent that produced it)
4. **Run agents per repo, each in its own subprocess** — `deps` always; `upstream` for
   repos with a configured upstream URL (`SWEEP_UPSTREAM_REMOTES`, since a fresh clone
   has no upstream remote). The subprocess env points the task store at THAT clone
   (`TASK_STORE_BACKEND=git-bug` + `GIT_BUG_TASK_STORE_REPO_PATH`), so advisories land in
   the swept repo and render in the sprinkles UI; `git-bug pull`/`push` run best-effort
   around the agents to keep dedupe accurate and publish filed advisories.
5. **Summarize** — one row per agent run (status parsed from the agent's own summary
   headline) plus a JSONL decision log.

**Fail isolation:** one repo failing never stops the sweep — clone failures and agent
crashes become rows/errors in the summary. The sweep exits non-zero (2) only for
driver-level failures: no host, enumeration failed, workdir unusable.

## Configuration

`SWEEP_` prefix (pydantic-settings): `HOST` (required, e.g. `code-public`), `PORT`
(default 23231), `WORKDIR` (default `~/.cache/forge-sweep`), `INCLUDE`/`EXCLUDE` (JSON
lists of globs), `UPSTREAM_REMOTES` (JSON object, repo name → upstream URL),
`DEPS_ENABLED`/`UPSTREAM_ENABLED`, `TASK_STORE_BACKEND` (default `git-bug`; `""` inherits
the caller's env), `BUG_USER_NAME`/`BUG_USER_EMAIL` (git-bug identity created in fresh
clones), timeouts, `AUTO_LOG_PATH`.

Agent-specific env passes straight through the sweep to the subprocesses — set
`UPSTREAM_SYNC_OPENAI_BASE_URL`/`_API_KEY` (collision seat) and `TASK_WORKER_SANDBOX`
(green-suite gate) alongside the `SWEEP_*` vars.

## Deployment

`forge-sweep.service` + `forge-sweep.timer` (nightly 04:47, before the dependabot
bumper's 05:37 run) — template units; install machine-local copies under
`~/.config/systemd/user/` with your real host and env file. `--auto-merge` is a
deliberate per-run flag and stays OFF by default; convergence is one bump per repo per
sweep by design.
