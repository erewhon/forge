# Coding Pipeline

Architect + orchestrator over Forge task trees: turn one **goal spec** into an
**epic** — a framed, human-approved, worker-shaped task tree — then drive it
through serial waves of the [task worker](../task_worker/README.md), verify
each wave, and end with a full-quorum sign-off that is **ready for a human
merge**. The pipeline never advances `main`.

Design doc: [`meta/coding-pipeline-design.md`](../../coding-pipeline-design.md).
Everything below was validated live on the toy-epic dry-run and the pipeline's
own fix-up epics (2026-07-03/04).

## Pattern

```
 A0 inventory → A1 framing ──[HUMAN approves]──→ A2 decompose → A3 emit to Forge
                                                                     │
        ┌────────────────────────── wave loop ───────────────────────┘
        │  reconcile orphans → plan wave (epic ref-prefix scope)
        │  → dispatch serially (worker, sandboxed, epic-context preamble)
        │  → verify (suite gate + review: collect → consolidate → confirm)
        │  → replan (deterministic escalations first; model for judgment)
        │  → persist wave record → [--wave-gate: stop for human review]
        └──→ dry (tree exhausted) │ waiting-on-human │ max-waves
                                                                     │
 epic gate: full-quorum cross-family sign-off on main..pipeline/<epic>
 → APPROVED renders the exact `jj` command for a HUMAN merge. Never merges.
```

**Human gates, in order:** framing approval (`--approve` refuses without it),
optional per-wave stop (`--wave-gate`), escalations (failed leaves land
`Spec Needed` + `Manual` at the attempt cap), and the terminal human merge.

## Quickstart (the toy-epic flow)

```bash
# 0. One-time per target repo: jj/git + a running `gaol dx` container
#    (see ../task_worker/README.md "Requirements per project"), and a Forge
#    project whose name matches the checkout dir (lowercased, no spaces).

# 1. Write a goal spec (.yaml, or .md with YAML frontmatter):
#      goal: "Add a temperature domain and a list-units subcommand"
#      project: "Pipeline-Smoke"        # Forge project, must exist
#      repo: /path/to/checkout          # optional; default = cwd
#      context: >- ...                  # constraints, prior art
#      value_hints: ["temperature ships first"]
#      epic_slug: smoke-tempconv        # optional; architect derives one

# 2. Frame (A0+A1) — writes pipeline-runs/<epic>/framing.{json,md}, then STOPS:
cd /path/to/checkout
meta build plan goal.yaml --project Pipeline-Smoke

# 3. Read framing.md. To approve: set "approved": true in framing.json.
#    Then decompose + emit the tree (A2+A3, idempotent by external_ref):
meta build plan goal.yaml --project Pipeline-Smoke --approve

# 4. Triage the tree in Forge (modes/tiers/caps), then run waves:
meta build run <epic-slug> --project Pipeline-Smoke --dry-run    # plan only
meta build run <epic-slug> --project Pipeline-Smoke --wave-gate  # 1 wave, stop
meta build run <epic-slug> --project Pipeline-Smoke --max-waves 3

# 5. When the loop reports dry (or the remaining leaves are yours):
meta build gate <epic-slug>      # full-quorum sign-off; renders the merge cmd
meta build status <epic-slug> --project Pipeline-Smoke

# 6. YOU merge:  jj bookmark set main -r pipeline/<epic-slug> && jj git push --bookmark main
```

`run` and `gate` operate on the repo at **cwd**; `plan` uses the spec's `repo`
field (falling back to cwd). Re-running `plan --approve` re-emits the tree
idempotently — but decomposition is model-driven and can legitimately *grow*
the tree on a re-run; expect "skipped (existing)" for old leaves, not 0 created.

## Epic membership & scope

An epic's leaves are everything whose `external_ref` starts with
`pipeline:<epic-slug>:` — tree leaves and replan fix-ups alike, whatever
Feature value they carry (decomposition legitimately spreads one epic across
several features). `--feature` narrows; tasks without a pipeline ref are out of
scope even in the same Feature. Fix-up refs key on the **finding's** slug
(`pipeline:<epic>:fix:<finding-slug>`) so re-discovered findings dedup across
replans.

## What a dispatched leaf gets

The dispatcher prepends an **epic-context preamble** to the worker's spec:
the landed interfaces of the leaf's direct dependencies (ast-extracted
signatures, public methods, module constants — read from the current tree, so
they cannot drift), plus titles-only fencing for the rest of the epic. Capped
at 4k chars, truncation announced, injection failures journal and dispatch
plain — context can never block a wave. Audit trail: `leaf_context` records in
the journal.

Per-leaf safety is the worker's (fresh gate re-check, clean-WC guard,
`max_files`, lint gate with autofix, tests, revert-on-fail, degenerate-session
retry) — see [`../task_worker/README.md`](../task_worker/README.md).

## Wave verification

- **Hard gate:** the whole-project suite runs in the dx container. Red = the
  wave does not advance; the tail feeds replan.
- **Advisory review:** each active pr_review provider lists findings over the
  wave diff → a **consolidation pass** merges paraphrase twins and drops
  candidates already covered by open fix-up leaves (fail-open: a down or
  inventing consolidator passes raw findings through, flagged) → a **confirm
  vote** (strict majority of responding skeptics) marks findings `confirmed`.
  Only confirmed findings can become fix-up leaves. The journal's review line
  shows the funnel: `N confirmed of M canonical (raw R, D covered by open fixups)`.

## Replan

Deterministic pre-rules never touch a model: a failed leaf at the attempt cap
(`max_leaf_attempts`, counted from the journal) escalates to `Spec Needed` +
`Manual` with diagnostics. The model is consulted only for judgment work
(confirmed findings → fix-ups, under-cap failures → respecs, integration-red
→ integration-fix leaf); a replan whose output fails validation **degrades**
to the deterministic escalations and journals `replan-degraded` — it never
kills the wave. (Model-replan output quality is tracked in
`pipeline:build:fix:replan-validation`.)

## Run directory & resumability

`pipeline-runs/<epic>/` (gitignored): `framing.{json,md}`, `inventory.{json,md}`,
`tree.{json,md}`, `wave-NNNN.json`, `journal.jsonl`. The journal is the source
of truth: leaf attempts are counted from `leaf_dispatch` records (they survive
crashes), wave numbering continues from the highest persisted `wave-*.json`,
and the orchestrator reconciles orphaned In Progress leaves at startup. To
continue an epic, just re-run `meta build run` — numbering and attempts resume.

## VCS & blast radius

All work lands on `pipeline/<epic-slug>` (created at the current tip, never
moved if it exists; advanced + pushed at each wave checkpoint — push failures
are warnings). `main` moves only by a human after the epic gate approves.
There is deliberately no auto-merge code path.

## Configuration (`CODING_PIPELINE_*` env vars)

| Var | Default | Meaning |
|---|---|---|
| `RUNS_DIR` | `~/Projects/erewhon/meta/pipeline-runs` | Per-epic run dirs |
| `MAX_WAVES` | `3` | Waves per *run* (re-run to continue) |
| `WAVE_SIZE` | `4` | Max leaves dispatched per wave |
| `MAX_LEAF_ATTEMPTS` | `2` | Journal-counted attempts before escalation |
| `BRANCH_PREFIX` | `pipeline` | Epic bookmark prefix |
| `INVENTORY_MAX_CHARS` | `40000` | A0 budget (tree-first trim, drops counted) |
| `INVENTORY_TREE_DEPTH` | `3` | A0 tree depth |
| `LLM_BACKEND` | `openai` | Headless architect backend (`openai`\|`anthropic`) |
| `OPENAI_BASE_URL` | `http://localhost:4010/v1` | Router endpoint |
| `OPENAI_API_KEY` | `sk-local-router` | Router key |
| `ARCHITECT_MODEL` | `coder` | Headless architect alias |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Anthropic-backend model |
| `ARCHITECT_MAX_TOKENS` | `8192` | Framing budget |
| `ARCHITECT_TIMEOUT` | `240` | Architect call timeout (s) |
| `DECOMPOSE_MAX_TOKENS` | `16000` | Tree budget |
| `DEFAULT_AUTO_MAX_FILES` | `5` | Cap for untagged Auto-* leaves (floor: 3 when tests required) |
| `LEAF_MODEL_TIER` | `coder` | Floor for unset/`auto` tiers on Auto-* leaves |
| `REVIEW_MAX_TOKENS` | `4096` | Review/sign-off budgets |
| `REVIEW_TIMEOUT` | `180` | Review call timeout (s) |
| `REVIEW_MAX_FINDINGS` | `12` | Candidate cap before consolidation |
| `CONFIRM_CONCURRENCY` | `4` | Parallel confirm votes |

Worker knobs (`TASK_WORKER_*`: sandbox, timeouts, degenerate-session retry,
commit prefix) are documented in [`../task_worker/README.md`](../task_worker/README.md).
The interactive strong-tier architect is [`/meta-architect`](../../../.claude/skills/)
— same stages, the session is the model; `meta build plan` is the headless path.

**Judgment evals** for the coding pipeline agents live in [`../evals/`](../evals/README.md) —
frozen gold-set grading that catches prompt regressions from distillation or model swaps.

## Operator playbook (what the dry-runs taught)

- **Escalated leaf** (`Spec Needed`+`Manual`): read the diagnostics note. Fix
  the spec / raise `max_files` / bump `model_tier`, then set Ready (+ Auto-*
  if you re-arm it for the worker) via `update_task_status`.
- **`waiting-on-human` exit** is normal, not an error: Manual, Spec Needed, or
  blocked leaves remain. The loop never spins on them.
- **Wave gate review:** read the wave record + diff; operator style/test
  cleanups commit directly on the epic branch; re-run `gate` after.
- **Degraded quorum at the gate** (a seat didn't respond) blocks fail-closed —
  re-run the gate.
- **A blocked gate with accurate blockers** usually means the epic really is
  half-done; finish the leaves rather than arguing with the reviewers.
