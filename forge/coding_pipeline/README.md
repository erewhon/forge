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
         │  → dispatch (serial by default; concurrent: fan-out → per-leaf workspaces → run-once sandboxes)
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

The dispatcher prepends an **epic-context preamble** to the worker's spec in both the serial
and concurrent paths: the landed interfaces of the leaf's direct dependencies (ast-extracted
signatures, public methods, module constants — read from the current tree, so they cannot drift),
plus titles-only fencing for the rest of the epic. Capped at 4k chars, truncation announced,
injection failures journal and dispatch plain — context can never block a wave. Audit trail:
`leaf_context` records in the journal.

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

A gitignored `.env` at the repo root is also read when running from the repo.

| Var | Default | Meaning |
|---|---|---|
| `RUNS_DIR` | `~/projects/pipeline-runs` | Per-epic run dirs |
| `MAX_WAVES` | `3` | Waves per *run* (re-run to continue) |
| `WAVE_SIZE` | `4` | Max leaves dispatched per wave |
| `MAX_LEAF_ATTEMPTS` | `2` | Journal-counted attempts before escalation |
| `DISPATCH_CONCURRENCY` | `3` | Bounded fan-out for concurrent dispatch (1 = serial, the old path) |
| `BRANCH_PREFIX` | `pipeline` | Epic bookmark prefix |
| `INVENTORY_MAX_CHARS` | `40000` | A0 budget (tree-first trim, drops counted) |
| `INVENTORY_TREE_DEPTH` | `3` | A0 tree depth |
| `LLM_BACKEND` | `openai` | Headless architect backend (`openai`\|`anthropic`) |
| `OPENAI_BASE_URL` | `http://localhost:4000/v1` | Router endpoint |
| `OPENAI_API_KEY` | `<your-router-key>` | Router key |
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
Concurrent dispatch uses a per-repo lockfile (`.task_worker/dispatch.lock`) to prevent
racing waves; see the [Concurrent dispatch](#concurrent-dispatch) section above.
The interactive strong-tier architect is [`/meta-architect`](../../../.claude/skills/)
— same stages, the session is the model; `meta build plan` is the headless path.

**Judgment evals** for the coding pipeline agents live in [`../evals/`](../evals/README.md) —
frozen gold-set grading that catches prompt regressions from distillation or model swaps.

## Concurrent dispatch

When `dispatch_concurrency` (env: `CODING_PIPELINE_DISPATCH_CONCURRENCY`, flag: `--concurrency`)
is **1**, leaves dispatch **serially** against the main working copy, one at a time — the
byte-for-byte pre-concurrency behaviour, kept as the escape hatch.

Above **1** the dispatcher runs a bounded-fan-out pattern:

```
  ready-set (N leaves)
        │
        ▼
  per-leaf jj workspaces (at shared base rev)
        │  (ephemeral gaol run-once sandboxes)
        ▼
  fan-out: cap K concurrent workers   ← dispatch_concurrency / --concurrency
        │
        ▼
  serial reconcile barrier
    • rebase each landed commit in dispatch order
    • conflict (jj first-class) → abandon + demote to Ready
    • jj infra failure → skip (don't abandon) + demote to Ready
        │
        ▼
  wave-level suite gate
        │
        ▼
  bisect-on-red (concurrent only, 2+ landed, suite red)
    → locate the first offending leaf, back it out, demote to Ready
        │
        ▼
  replan (demoted leaves flow into the existing machinery)
```

### The two kinds of independence

- **DAG gives logical independence.** Dependency edges in the epic tree define a topological
  order — leaves without unresolved deps are *logically* safe to run in any order.
- **The barrier enforces physical independence.** Concurrent dispatch ignores the DAG beyond
  topological readiness; the serial reconcile barrier catches the physical conflicts that the
  DAG can't predict (unrelated files touching shared behaviour, test fixtures colliding, etc.).
  Correctness rests entirely on the barrier.
- **`file_scope` only optimises.** The scheduler (scheduling.py) reads architect-predicted
  file-scopes from the tree and greedily batches leaves with disjoint scopes, deferring
  overlapping ones to the next wave. A wrong pick costs wasted parallel work (a colliding
  leaf burns a worker run before detection reverts it), never corruption. The picker only
  engages when the tree carries scope data at all: with scopes present, a scope-less leaf
  (replan fix-ups carry none) is unknown territory and dispatches alone; with NO scope data
  anywhere (legacy tree, no tree) the wave stays fully optimistic — prediction must never
  become a gate. This epic shipped the file-scope hint emission through the tree (LeafSpec)
  and the conflict-free batch picker; it is a pure optimisation layered on top of the
  barrier.

### `dispatch_concurrency` and `--concurrency`

| Setting | Behaviour |
|---|---|
| `dispatch_concurrency = 1` | Serial path — every leaf runs one at a time on the main working copy. The pre-concurrency behaviour exactly. |
| `dispatch_concurrency = K > 1` (default: 3) | Bounded fan-out: at most K leaves run concurrently, each in its own jj workspace under an ephemeral `gaol run-once` sandbox. Integrates through the serial reconcile barrier. |

The `--concurrency N` flag on `meta build run` overrides the config default for that invocation.
The default is **3** (raised from 1 after the deliberate-conflict smoke passed on 2026-07-07:
a colliding pair was dispatched; one leaf landed cleanly, the other was detected, demoted,
and retried onto the updated head by the reconcile barrier). The full evidence record —
leaf specs, run transcripts, journal excerpts, end-state verification, and the reproduction
runbook — is checked in at
[`examples/cw-smoke-2026-07-07.md`](examples/cw-smoke-2026-07-07.md); the live smoke is a
Manual leaf by design (it needs the router, gaol, and Forge), so it is a recorded run, not
a suite test.

### Conflict demotion semantics

These invariants were proven live by the deliberate-conflict end-to-end smoke (2026-07-07):

- **Done-then-Ready flips carry notes.** A leaf that lands Done in its workspace but conflicts
  during the reconcile barrier is demoted back to `Ready` with a note explaining the conflict
  paths and that the commit was abandoned. The demotion is a *separate* journal event
  (`reconcile_demotion`), not a second `leaf_dispatch`, so the attempt cap is never double-counted.
- **Bisect-on-red.** If the concurrent wave's batch suite goes red but no leaf conflicted at
  the barrier (a semantic conflict: green in isolation, red combined), a linear walk over the
  integrated chain locates the first offending leaf, backs it out, and demotes it. Single-leaf
  waves and serial waves already have leaf-level attribution — bisect is concurrent-only.
- **Serial-fallback guarantee.** `dispatch_concurrency = 1` is the escape hatch: it reproduces
  the pre-concurrency serial path exactly. Use it when debugging, when the suite is too noisy,
  or when running on repos that don't support jj workspaces.

### Repo lock

A per-repo lockfile (`.task_worker/dispatch.lock`, pid inside) prevents two dispatch waves
from racing on the same working copy. A live holder raises `DispatchError` and aborts the
wave; a dead holder's lock is stolen. The lock covers workspace creation, fan-out, and
reconcile — the entire dispatch critical section.

### Workspaces and sandboxes

Each leaf gets its own jj workspace created at the wave's shared base revision. Workers run
under `gaol run-once` sandboxes (ephemeral — they don't persist across invocations) rather
than the path-bound dx container. After dispatch (success, conflict, or crash) all workspaces
are forgotten, so the main repo is the only persistent artefact.

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
