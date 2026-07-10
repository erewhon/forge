# meta deps

Dependency bumper: scan, bump, gate, and auto-merge clean low-risk bumps.

## What it does

`meta deps` runs the scan → bump → gate → act loop on the current repository:

1. **Scan** — finds outdated direct dependencies via `uv`
2. **Audit** — runs pip-audit against the current lockfile
3. **Pick** — selects the first patch/minor candidate (major bumps go straight to advisory)
4. **Bump** — applies `uv lock -P <name>` for the candidate (manifest+lockfile only)
5. **Gate** — three checks, all must pass for auto-merge:
   - **Manifest-only** — only dependency manifests/lockfiles changed
   - **Green suite** — the test suite passes against the bumped lock
   - **Supply-chain sign-off** — full-quorum approval from the review ensemble's provider roster
6. **Act** — pushes a `deps/<name>-<version>` branch; with `--auto-merge`, advances `main`

Policy-ineligible candidates and gate misses fall through to an advisory branch plus a Forge
task, never a merge (fail-closed).

## Threshold policy

- **Auto-merge branch**: manifest-only diff, green suite, and full-quorum supply-chain sign-off
  — only for clean patch or minor bumps.
- **Advisory branch**: everything else (major bumps, any gate miss, missing risk evidence, or a
  degraded quorum) — pushed as a branch with a Forge task filed for human review.

## Decision log

Decisions are appended as JSONL records to `forge/dependabot/logs/auto.jsonl` in the agent's
package directory.

## Honest scope — what the evidence does and does NOT cover

The sign-off judges **metadata**: audit findings, version delta, release age, yanked flag,
changelog presence, typosquat distance, the lockfile delta, and (v2) two provenance signals
compared across the bump itself:

- **Maintainer-identity change** — the current and target releases' author/maintainer
  fields (emails primarily, names as fallback) are compared; a difference blocks the auto
  track. Stateless and scoped to the exact bump window; deeper history would need an
  external feed.
- **New install/build scripts** — both releases' sdists are downloaded (size-capped, read
  in memory, never executed): a target that introduces a top-level `setup.py` or changes
  its build backend blocks the auto track.

Both signals are best-effort: when they cannot be determined they report `unknown`, which
never blocks on its own and never marks the evidence incomplete. What v1+v2 still do NOT do
is diff the dependency's source: a compromised release with stable metadata, an unchanged
build backend, and no new scripts can pass — the conservative dial (patch/minor only,
complete evidence, unanimous cross-family sign-off, one bump per branch) is the mitigation,
not a detection claim.
