# forge upstream

Upstream sync for additive forks: fetch, merge, gate, push — fail-closed.

## What it does

`forge upstream` keeps a fork current with its upstream remote:

1. **Fetch** — `git fetch <remote>` (default remote: `upstream`)
2. **Compare** — merge-base vs the upstream default branch; up to date → stop
3. **Layer** — compute the fork's additive layer from the merge-base (files the fork
   added, upstream files it modified) — the collision seat's ground truth
4. **Merge** — `git merge --no-ff` on an `upstream-sync/<date>-<tip>` branch inside a
   **disposable worktree**; the caller's checkout is never touched (a dirty working copy —
   or a jj-colocated repo whose working copy jj owns — is never at risk)
5. **Gate** — two checks:
   - **Green suite** — the repo's tests in the task worker's sandbox
   - **Collision seat** — an LLM judges whether upstream changed something the layer
     imports, wraps, or hooks (the semantic breakage a clean textual merge hides).
     Findings must cite a file from the evidence; uncited concerns demote to notes.
6. **Act** — pushes the sync branch (even on a gate miss — the reviewing human wants the
   branch); `--auto-merge` advances the remote default branch only when the suite is green,
   the seat affirmatively says no collision (unknown blocks), and local main matches
   origin's

A textual merge conflict files an advisory task listing the conflicted files and pushes
nothing — there is no mergeable state to push. Gate misses push the branch AND file the
task. Advisory tasks dedupe by upstream tip, so re-runs against the same upstream state
never stack tasks.

## Configuration

`UPSTREAM_SYNC_` prefix (pydantic-settings): `REMOTE` (default `upstream`),
`UPSTREAM_BRANCH`/`LOCAL_BRANCH` (auto-detect main/master), `BRANCH_PREFIX`,
`SEAT_ENABLED`/`SEAT_MODEL`/`OPENAI_BASE_URL`/`OPENAI_API_KEY` (the seat's router),
`AUTO_LOG_PATH` (JSONL decision log).

With `TASK_STORE_BACKEND=git-bug` pointed at the fork, advisory tasks are filed INTO the
fork as git-bug issues — they travel with the repo and render in the Soft Serve sprinkles
UI.

## Deployment notes

- The green-suite gate runs in the task worker's sandbox. gaol **dx** containers are keyed
  by directory name, and the temporary worktree never has one — set
  `TASK_WORKER_SANDBOX=gaol-run-once` for real deployments, or every sync rides the
  advisory track with a "container not found" gate failure (fail-closed, but noisy).
- v1 is git-backed repos (jj-colocated works — the agent only adds refs; a pure jj repo
  without `.git` does not).
- `--auto-merge` advances the REMOTE default branch; the local branch is deliberately left
  behind for the owner to pull — the agent never rewrites a checkout it doesn't own.
