# forge sweep

Fleet sweep: run the per-repo agents across every repo on a Soft Serve instance or a set
of GitHub owners.

## What it does

1. **Enumerate** — `SWEEP_SOURCE=soft-serve` (default): `ssh -p <port> <host> repo list`;
   `SWEEP_SOURCE=github`: `gh repo list <owner>` for each of `SWEEP_GITHUB_OWNERS`,
   skipping archived repos and forks by default (forks are `upstream`'s job)
2. **Filter** — fnmatch include/exclude globs on repo names
3. **Clone/refresh** — a machine-owned workdir of clones: clone on first sight, `fetch` +
   `reset --hard origin/<default>` on later sweeps (the workdir is a cache; anything of
   value was pushed by the agent that produced it)
4. **Run agents per repo, each in its own subprocess** — `deps` for every repo with a
   supported ecosystem (`uv.lock`/`go.mod`; anything else is a benign `skipped` row until
   more ecosystem adapters exist); `upstream` for repos with a configured upstream URL
   (`SWEEP_UPSTREAM_REMOTES`, since a fresh clone has no upstream remote). The subprocess
   env points the task store at THAT repo: git-bug advisories in the clone for Soft Serve
   (they travel with the repo and render in the sprinkles UI; `git-bug pull`/`push` run
   best-effort around the agents), GitHub Issues (`GITHUB_TASK_STORE_REPO=<owner/repo>`)
   for the github source.
5. **Summarize** — one row per agent run (status parsed from the agent's own summary
   headline) plus a JSONL decision log.

**Fail isolation:** one repo failing never stops the sweep — clone failures and agent
crashes become rows/errors in the summary. The sweep exits non-zero (2) only for
driver-level failures: no host, enumeration failed, workdir unusable.

## Configuration

`SWEEP_` prefix (pydantic-settings): `SOURCE` (`soft-serve`/`github`), `HOST` + `PORT`
(soft-serve source), `GITHUB_OWNERS` (JSON list, github source) with
`SKIP_ARCHIVED`/`SKIP_FORKS` (default true) and `CLONE_PROTOCOL` (`https` default — gh's
credential helper serves it; `ssh` for keys), `WORKDIR` (default `~/.cache/forge-sweep`),
`INCLUDE`/`EXCLUDE` (JSON lists of globs), `UPSTREAM_REMOTES` (JSON object, repo name →
upstream URL), `DEPS_ENABLED`/`UPSTREAM_ENABLED`, `TASK_STORE_BACKEND` (`auto` follows
the source: git-bug for soft-serve, github issues for github; `inherit` leaves the
caller's env untouched; or an explicit backend name), `BUG_USER_NAME`/`BUG_USER_EMAIL`
(git-bug identity created in fresh clones), timeouts, `PRUNE` (or `--prune`: remove
workdir clones absent from the FULL enumeration — for repo shuffles; a scoped `--only`
run never prunes what it didn't look at), `AUTO_LOG_PATH`.

Agent-specific env passes straight through the sweep to the subprocesses — set
`UPSTREAM_SYNC_OPENAI_BASE_URL`/`_API_KEY` (collision seat) and `TASK_WORKER_SANDBOX`
(green-suite gate) alongside the `SWEEP_*` vars.

## Deployment

`forge-sweep.service` + `forge-sweep.timer` (nightly 04:47, before the dependabot
bumper's 05:37 run) — template units; install machine-local copies under
`~/.config/systemd/user/` with your real host and env file. `--auto-merge` is a
deliberate per-run flag and stays OFF by default; convergence is one bump per repo per
sweep by design.
