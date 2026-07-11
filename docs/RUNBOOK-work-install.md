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

## Models & executors

**Router.** Forge's tiers (`coder`, `auto`, `auto-free`, `auto-full`, `research`) resolve through
any OpenAI-compatible endpoint. On the work Mac, run LiteLLM in the foreground with the committed
recipe — Bedrock serves the strong tiers, a local model on the M3 serves the cheap tier:

```bash
pip install 'litellm[proxy]'                    # or: uv tool install 'litellm[proxy]'
export AWS_REGION=us-east-1                     # + standard AWS credential chain (env/profile/SSO)
litellm --config docs/work/litellm-work.yaml --port 4000
```

Port 4000 matches forge's neutral `http://localhost:4000/v1` defaults, so no base-url env vars
are needed while the router is up. Bedrock model access must be enabled in the AWS account —
verify with `aws bedrock list-foundation-models --by-provider anthropic`.

**Work `.env`** (repo root, gitignored — or plain env vars):

```bash
TASK_STORE_BACKEND=github
GITHUB_TASK_STORE_REPO=owner/repo
# Only needed if the proxy is keyed (LITELLM_MASTER_KEY set on the router):
# CODE_REVIEWER_OPENAI_API_KEY=<router key>   # one per agent prefix in use
```

**Executor: OpenCode first.** Bedrock bills per token either way, so OpenCode driving the local
router is the identical wiring to the home setup. Point opencode at the router in
`~/.config/opencode/opencode.json`:

```json
{
  "provider": {
    "llm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "LLM Router (work)",
      "options": { "baseURL": "http://localhost:4000/v1", "apiKey": "none" },
      "models": { "auto": {}, "auto-free": {}, "auto-full": {}, "coder": {}, "research": {} }
    }
  }
}
```

*Alternative:* Claude Code natively supports Bedrock — `export CLAUDE_CODE_USE_BEDROCK=1` with AWS
credentials makes `claude -p` executor seats work without the router. Documented for completeness;
OpenCode-through-router is the default work path.

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
| Router up: litellm serves the five tiers, Bedrock reachable | ⬜ pending — needs the macOS work machine |
| Executor smoke: `forge edit` via opencode against a Bedrock tier | ⬜ pending — needs the macOS work machine |
| Cheap tier: M3-local model selected via `forge evals` and pinned | ⬜ pending — needs the macOS work machine |
| `forge task` picks leaf → Apple Container sandbox run → GitHub issue status update | ⬜ pending — needs the macOS work machine |

When running the final check on the Mac: create a scratch private repo, file one ready issue in the
GitHub task-store format, run `forge task --dry-run` first, then the real run, and confirm the
issue's `status:*` label advanced.
