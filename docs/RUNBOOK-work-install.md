# Runbook: work install (GitHub task backend, no Nous)

Install and run forge on a locked-down machine: GitHub issues as the task store, no Nous anywhere,
agents sandboxed by [gaol](https://github.com/erewhon/gaol) dx.

## Install

```bash
# ephemeral (nothing on PATH):
uvx --from 'git+https://github.com/erewhon/forge' forge --help

# persistent tool (HTTP git access is enough — no GitHub API needed to install):
uv tool install 'git+https://github.com/erewhon/forge'      # `forge` + `forge-mcp`
```

Neither form installs `nous_ai`/`nous_mcp` — the Nous backend lives behind the `[nous]` extra.

## Configure

Settings are pydantic-settings: env vars, or a `.env` in the directory you run from (each agent has
its own prefix — `TASK_WORKER_`, `CODE_REVIEWER_`, `EVALS_`, `CODING_PIPELINE_`, etc.):

```bash
export TASK_STORE_BACKEND=github
export GITHUB_TASK_STORE_REPO=owner/repo        # issues live here
export <PREFIX>_OPENAI_BASE_URL=http://<router>/v1
export <PREFIX>_OPENAI_API_KEY=<key>
```

Selecting `TASK_STORE_BACKEND=forge` (the Nous backend) without the extra fails fast with an
install hint — that's the guard working.

## Run

```bash
forge task --project <Project> --dry-run    # pick a ready leaf, execute, revert (rehearsal)
forge task --project <Project>              # the real thing: execute + commit + status back
```

The worker executes leaves inside the gaol dx sandbox. On macOS the sandbox backend is Apple
Container — see the Gaol project's "Apple Container Runtime Integration" task for that runtime;
this runbook does not reimplement it.

## Verification checklist

| Check | Status |
|---|---|
| Clean `uvx --from git+…` install, `forge --help` exits 0 | ✅ verified 2026-07-10 (Linux, HEAD e4f4af2c) |
| `nous_ai` / `nous_mcp` absent without `[nous]` | ✅ verified 2026-07-10 |
| Config defaults neutral without a `.env` | ✅ verified 2026-07-10 |
| `TASK_STORE_BACKEND=github` + repo → `GitHubTaskStore` constructs | ✅ verified 2026-07-10 |
| Missing `GITHUB_TASK_STORE_REPO` → clear ValueError | ✅ verified 2026-07-10 |
| Nous backend without extra → actionable `forge[nous]` hint | ✅ verified 2026-07-10 |
| `forge task` picks leaf → Apple Container sandbox run → GitHub issue status update | ⬜ pending — needs the macOS work machine |

When running the final check on the Mac: create a scratch private repo, file one ready issue in the
GitHub task-store format, run `forge task --dry-run` first, then the real run, and confirm the
issue's `status:*` label advanced.
