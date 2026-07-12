# meta deps

Dependency bumper: scan, bump, gate, and auto-merge clean low-risk bumps.

## What it does

`meta deps` runs the scan → bump → gate → act loop on the current repository:

1. **Scan** — finds outdated direct dependencies
2. **Audit** — runs a known-vulnerability scan repo-wide
3. **Pick** — selects the first patch/minor candidate (major bumps go straight to advisory)
4. **Bump** — applies the candidate as a manifest+lockfile-only working-copy change
5. **Gate** — three checks, all must pass for auto-merge:
   - **Manifest-only** — only dependency manifests/lockfiles changed
   - **Green suite** — the test suite passes against the bumped lock
   - **Supply-chain sign-off** — full-quorum approval from the review ensemble's provider roster
6. **Act** — pushes a `deps/<name>-<version>` branch; with `--auto-merge`, advances `main`

Policy-ineligible candidates and gate misses fall through to an advisory branch plus a Forge
task, never a merge (fail-closed).

## Ecosystems

Scan/bump/delta/audit/evidence live behind an adapter port (`ecosystems/`); the loop, gates,
VCS actions, and advisory emission are ecosystem-neutral. Detection is by manifest —
`uv.lock` → uv, `go.mod` → go, `pnpm-lock.yaml` → pnpm, `Cargo.lock` → cargo; a repo
with several requires
`--ecosystem` (or `DEPENDABOT_ECOSYSTEM`) — the fleet sweep instead runs once per present
ecosystem. Backends:

- **uv (Python)** — `uv tree --outdated` / `uv lock -P <name>` / pip-audit, plus the full
  PyPI evidence bundle (yanked, release age, typosquat, maintainer change, install scripts,
  Scorecard, PEP 740 attestation).
- **go** — `go list -u -m -json all` / `go get <mod>@<ver>` + `go mod tidy` /
  `govulncheck -scan package -json ./...` (osv-scanner fallback; neither installed fails
  closed). There is **no Go-native provenance source wired yet**, so evidence is incomplete
  by construction and every Go bump rides the advisory track — with
  `TASK_STORE_BACKEND=git-bug` the advisory files straight into the target repo. Two Go
  behaviors to expect on the branch: minimum-version-selection ripple (bumping a module can
  raise its own requirements' minimums), and a one-time `go mod tidy` cleanup on repos whose
  go.mod carried stale `// indirect` annotations. The `--redundancy-report` sub-mode remains
  uv-only.
- **pnpm (JS/TS)** — `pnpm outdated --json` / `pnpm update <name> --ignore-scripts` (the
  security floor: dependency lifecycle scripts never execute on the host) / `pnpm audit`
  (osv-scanner fallback). The locked version comes from pnpm-lock.yaml's root importer
  (workspace members are audited but not bumped — v1 limitation); `pnpm update` rewrites
  the package.json range Dependabot-style, and both files are manifests. No npm-native
  provenance source yet, so pnpm bumps ride the advisory track by construction.
- **cargo (Rust)** — scan via the crates.io sparse index (true latest incl. out-of-constraint
  majors; `cargo update --dry-run` was rejected: whole-graph, stderr-only, in-constraint) /
  `cargo update -p <name>@<current>` (precise pkgid, lock-only) / cargo-audit (RUSTSEC;
  osv-scanner fallback). Direct deps come from the root Cargo.toml including
  `[workspace.dependencies]`; member manifests are not scanned (v1). Keys on `Cargo.lock` —
  libraries that don't commit a lockfile read as sweep skips. No crates.io evidence pass
  yet, so Cargo bumps ride the advisory track by construction.

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

---

## Provenance signals (v2)

The v2 signals live as optional fields on ``EvidenceBundle``. All are **best-effort**: when they
cannot be determined they report ``None``, which **never** marks the evidence incomplete and
**never** blocks on its own. The policy gates only when a signal is provably-True (or provably
below a threshold for scorecard).

| Signal | Evidence field | What it checks | Gate behaviour | None semantics |
|---|---|---|---|---|
| Scorecard floor | ``scorecard_score`` | OpenSSF Scorecard aggregate score for the package's source repo | **Blocks** when ``score < SCORECARD_FLOOR`` (default 5.0). Advisory + Forge task. | Best-effort pass — missing data does not block |
| Attestation posture | ``target_attested`` | PEP 740 provenance attestation presence on PyPI (via the integrity API) | **Blocks** when ``require_attestation=True`` and the target is not provably attested (``False`` or ``None``). Advisory + Forge task. | Treated as **unattested** when ``require_attestation=True`` — the auto track pauses, bumps still flow as advisory tasks |
| Maintainer change | ``maintainer_changed`` | Author/maintainer identity differs between current and target release (email comparison primary, names fallback) | **Blocks** when provably different. Advisory + Forge task. | Best-effort pass |
| Install scripts | ``new_install_scripts`` | Target sdist introduces a top-level ``setup.py`` or changes the build backend | **Blocks** when provably true. Advisory + Forge task. | Best-effort pass |
| Typosquat | ``typosquat_suspect`` | Package name is one edit away from a popular package (OsaDistance ≤ 2 against a curated list) | **Blocks** when set. Advisory + Forge task. | N/A — the field is ``None`` (not suspect) when no match |
| Reachability demotion | ``reachable`` | Whether the package is actually imported by this repo (AST import-graph analysis) | **Demotes** — packages provably not imported (``False``) receive lower advisory priority. Never blocks or promotes. | Unknown — treated as normal priority |

### Attestation posture: the ``None`` divergence

The attestation signal diverges from the other best-effort v2 signals in its handling of
``None``. When ``require_attestation=True`` (the default), ``None`` (undeterminable due to an
API outage or malformed response) is treated as **unattested**, because the auto track should
pause rather than guess. This is the correct failure mode for an integrity check.

All other v2 signals (scorecard, maintainer change, install scripts, reachability) treat
``None`` as **best-effort pass** — missing data does not become a block.

### ``--redundancy-report``

The ``--redundancy-report`` flag runs a **read-only** sub-mode: it scans the repo's direct
dependencies, asks a configured LLM to identify overlapping-purpose clusters, and prints a
markdown report to stdout. No bumps are applied, no branches pushed, no tasks emitted.

```bash
# Default: uses the local LiteLLM router (alias ``coder``)
forge deps --redundancy-report

# Specify a repo
forge deps --redundancy-report --repo /path/to/project
```

## Out of scope — deferred deliberately (deps-v2 framing)

Two sub-features of the original "deps v2 — supply-chain provenance" brief were re-homed
rather than built here, per the approved epic framing:

- **AI-BOM (model provenance inventory).** The models it would inventory belong to the LLM
  router, not this repo — filed in the **LLM Router** project as "Static CycloneDX ML-BOM for
  the router models" (task ref `deps-v2:ai-bom-moved`).
- **AI vendoring of small unmaintained dependencies.** Reimplementing a dependency writes
  code with wide design latitude — a response playbook, not an evidence signal. Deferred to
  its own future epic; `--redundancy-report` output is its natural candidate list.
