# forge

Front door for the erewhon code agents — a `forge` CLI and a `forge-mcp` MCP server over a fleet
of coding agents: review ensembles, parallel model comparison, research harnesses, dependency
bumping, testing loops, and an architect/worker coding pipeline.

`forge <verb> [args]` — run `forge <verb> --help` for each agent's own options.

- **Where tasks go:** verbs that emit (`audit`/`testing`/`refactor`/`deps`, the pipeline) file into
  the configured task store — a [Nous](https://github.com/erewhon/nous) Forge notebook (needs the
  `[nous]` extra), or GitHub issues with `TASK_STORE_BACKEND=github`.
- **Models:** the review/analysis ensembles run against any OpenAI-compatible endpoint (point
  `*_OPENAI_BASE_URL`-style settings at your router); `edit` and `task`/`build` can also drive
  Claude/OpenCode per their model settings.
- **MCP:** `research`, `review`, `book` are exposed as MCP tools via `forge-mcp`; the rest are
  CLI-only.

## Research & writing

| Example | Does |
|---|---|
| `forge research "why did the Bronze Age collapse?"` | Iterative research: plan → research → verify → synthesize (`--dry-run`, `--max-sprints N`) |
| `forge book config.yaml` | Book-length research via generator–evaluator sprint cycles |

## Review & analysis  (read-only)

| Example | Does |
|---|---|
| `forge review --pr 123 --repo owner/repo` | PR review ensemble; `--post-comment` posts back; `--pass digest\|supply-chain`; or `--diff-file x.diff` |
| `forge audit crates/foo/src --project Gaol --emit-tasks` | Adversarial multi-model audit (discover→dedup→verify); `--focus "data loss"`, `--min-severity high`, `--dry-run-emit` |
| `forge testing crates/foo/src --project Gaol --emit-tasks` | Find untested behavior → file test-gap tasks (`--auto` instead *generates+gates+pushes* tests) |
| `forge refactor crates/foo/src --project Gaol --emit-tasks` | Find smells, verify safe+worthwhile, file refactor tasks |
| `forge code-review` | Nightly code review of recent commits (markdown by default; optional Nous daily-note sink) |

## Code generation & autonomy

| Example | Does |
|---|---|
| `forge edit --prompt "convert callbacks to async" --repo ~/src/foo --models "claude-opus-4-8,opencode:glm-5.1"` | Same prompt, N models (2–26), compare the diffs |
| `forge task --project Gaol` | Autonomous worker: pick the top ready task, run it in the sandbox, commit (`--dry-run` executes then reverts) |
| `forge build plan --epic <slug> --project Meta --repo <path>` | Coding pipeline A0+A1 framing; `--approve` unlocks decompose + emit |
| `forge build run --epic <slug> --project Meta --repo <path>` | Orchestrator wave loop (dispatch → verify → replan); `--concurrency N` |
| `forge build gate --epic <slug> --repo <path>` | Final full-quorum epic sign-off (a human merges after) |
| `forge deps --project Meta --dry-run` | Dependency bumper: scan + gate; `--auto-merge` advances main on clean low-risk bumps |

## Eval

| Example | Does |
|---|---|
| `forge evals run` | Score models against frozen gold sets and print the report |
| `forge evals baseline` / `compare` | Save a scorecard as baseline / compare a fresh one against it |

## Install

```bash
uv sync                     # library + CLI, no Nous
uv sync --extra nous        # + the Nous task-store backend

# or as a global tool:
uv tool install --editable ~/path/to/forge          # `forge` + `forge-mcp`
uv tool install --editable '~/path/to/forge[nous]'  # with the Nous backend
```

Machine-local settings (router URL, API key, project paths) live in a gitignored `.env` at the
repo root — every agent's settings read it when run from the repo.

---

**Conventions.** Path-based verbs (`audit`/`testing`/`refactor`) read files/dirs into one context
(no file tools) — scope them to a crate/module, not a whole repo. First pass any emit with
`--dry-run-emit` to preview volume. Emitted tasks land **Spec Needed + Manual** (proposals),
deduped by a stable `external_ref`, so re-running the same sweep never duplicates.

## License

Apache-2.0
