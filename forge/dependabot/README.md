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

Decisions are appended as JSONL records to `agents/dependabot/logs/auto.jsonl` in the agent's
package directory.

## Honest scope

The evidence bundle is collected at the metadata level: pip-audit CVE findings, yank status,
package age, changelog availability, and typosquat suspicion. This agent does **not** perform
dependency source diff analysis — it reviews metadata, risk evidence, and manifest changes only.

## Usage

```
meta deps                  # scan, gate, and push (auto-merge OFF by default)
meta deps --dry-run        # plan only — no writes, no gates
meta deps --auto-merge     # also advance main when every gate passes
meta deps --repo /path     # run against a specific repo root
meta deps --project Foo    # file advisory tasks into the "Foo" Forge project
```

Exit codes: 0 = merged / branched / planned / no-candidates; 1 = advisory / error (scriptable).

## Honest scope — what the evidence does NOT cover (v1)

The sign-off judges **metadata**: audit findings, version delta, release age, yanked flag,
changelog presence, typosquat distance, and the lockfile delta. Two signals named in the
original brief are **deliberately descoped** in v1 because PyPI's JSON API cannot provide them:

- **Maintainer/owner change** — PyPI does not expose versioned maintainer history via the JSON
  API; detecting a handover needs an external feed or a stored per-package baseline.
- **New install/build scripts** — requires downloading and inspecting the sdist/wheel, which is
  source-level analysis, not metadata.

Both are v2 candidates (tracked in Forge: "Dependabot: maintainer-change and install-script
evidence signals"). Until then the conservative dial compensates: patch/minor only, complete
evidence required, unanimous cross-family sign-off, one bump per branch.
