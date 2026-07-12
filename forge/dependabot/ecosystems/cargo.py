"""The Cargo ecosystem adapter — Rust repos scanned via crates.io, bumped via cargo.

CLI/API surfaces pinned by live capture on 2026-07-12 (cargo 1.95.0, cargo-audit via brew,
fleet clones):

- **Scan is registry-index-based, NOT ``cargo update --dry-run``**: the dry-run prints to
  STDERR, updates the whole graph (56 packages on the live capture, transitive included),
  and only surfaces in-constraint updates. The sparse index
  (``https://index.crates.io/{prefix}/{name}``) is ndjson, publish-ordered, one line per
  version with ``vers``/``yanked`` — latest = max non-yanked parseable version, so
  prereleases (unparseable as int tuples) drop out naturally, and out-of-constraint majors
  ARE surfaced (uv-scan parity).
- **Direct deps come from the root Cargo.toml**: ``[dependencies]``,
  ``[dev-dependencies]``, ``[build-dependencies]``, and ``[workspace.dependencies]`` —
  modern workspaces centralize there (live capture: protectinator's root
  ``[dependencies]`` is empty, everything lives in the workspace table). Entries are
  version strings or tables; ``package = "real-name"`` renames are honored; path/git-only
  deps (no ``version`` key) are skipped — they have no registry latest. Workspace MEMBER
  manifests are not scanned (v1 limitation, like pnpm's root importer).
- **Current versions come from Cargo.lock** (``[[package]]`` name/version/source, parsed
  with tomllib): registry-sourced entries only; a direct crate locked at MULTIPLE versions
  is skipped — an ambiguous current is not a candidate (the pnpm unprovable-current rule).
- **Apply**: ``cargo update -p name@current`` (precise pkgid) — updates within the
  manifest constraint, lock-only change. An out-of-constraint latest leaves the lock
  unchanged → the constraint-pinned skip path, exactly like ``uv lock -P``.
- **Audit**: ``cargo-audit audit --json`` (exit 1 = vulnerabilities found);
  ``vulnerabilities.list[]`` carries ``advisory.id`` (RUSTSEC)/``title``/``aliases`` plus
  ``versions.patched`` as ``>=`` ranges — lower bounds are the fix versions.
  ``warnings`` (unmaintained/unsound/yanked) are v1-ignored. osv-scanner is the fallback;
  neither installed fails CLOSED.

Evidence posture: no crates.io evidence pass is wired yet, so ``collect_evidence`` returns
``complete=False`` with an explicit ``incomplete_reason`` — every Cargo bump rides the
advisory track by construction. The index's ``yanked`` flag and crates.io's ``created_at``
are the named follow-up that would open the Rust auto track.
"""

from __future__ import annotations

import json
import re
import subprocess
import tomllib
from pathlib import Path

import httpx

from forge.dependabot.audit import AuditError
from forge.dependabot.bump import BumpError, _parse_lockfile_diff
from forge.dependabot.config import settings
from forge.dependabot.ecosystems.pnpm import _PATCHED_LOWER_RE
from forge.dependabot.models import AuditFinding, BumpCandidate, EvidenceBundle
from forge.dependabot.scan import _DELTA_ORDER, classify_delta
from forge.dependabot.supply_chain import _version_tuple
from forge.shared.automerge import working_diff
from forge.task_worker.vcs import VCSError, get_changed_files, revert_changes

_CARGO_INCOMPLETE_REASON = (
    "no crates.io provenance evidence source yet (yanked/release-age/attestation/Scorecard "
    "signals are PyPI-only) — Cargo bumps ride the advisory track by construction"
)

_INDEX_URL = "https://index.crates.io/{prefix}/{name}"

# Cargo.lock hunk header in a git/jj unified diff (the block parser is shared with uv.lock).
_CARGO_LOCK_HEADER_RE = re.compile(r"^(?:---|\+\+\+) .*/?Cargo\.lock")


class CargoEcosystem:
    name = "cargo"

    # --- scan ---------------------------------------------------------------------------

    def scan_outdated(self, repo: Path, *, timeout: float | None = None) -> list[BumpCandidate]:
        """Direct registry deps whose crates.io latest exceeds the locked version.

        One sparse-index request per direct dep (bounded by the manifest, ~tens). A dep
        whose index fetch fails is dropped for this run — a candidate is never fabricated
        from partial data; the next sweep retries.
        """
        timeout = timeout if timeout is not None else settings.metadata_timeout
        direct = _direct_deps(repo)
        locked = _locked_versions(repo)
        candidates: list[BumpCandidate] = []
        for crate in sorted(direct):
            current = locked.get(crate)
            if not current:
                continue
            latest = _index_latest(crate, timeout=timeout)
            if not latest or latest == current:
                continue
            candidates.append(
                BumpCandidate(
                    name=crate,
                    current=current,
                    latest=latest,
                    delta=classify_delta(current, latest),
                )
            )
        candidates.sort(key=lambda c: (_DELTA_ORDER[c.delta], c.name))
        return candidates[: settings.max_candidates]

    # --- apply --------------------------------------------------------------------------

    def apply_bump(
        self, repo: Path, candidate: BumpCandidate, *, timeout: int | None = None
    ) -> list[str]:
        """``cargo update -p name@current``; revert + BumpError on failure.

        The precise pkgid pins WHICH lock node updates; cargo moves it as far as the
        manifest constraint allows (an out-of-constraint latest → unchanged lock → the
        caller's constraint-pinned skip)."""
        timeout = timeout if timeout is not None else settings.audit_timeout
        result = subprocess.run(
            ["cargo", "update", "-p", f"{candidate.name}@{candidate.current}"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            revert_changes(repo)
            raise BumpError(
                f"cargo update -p {candidate.name}@{candidate.current} failed "
                f"(exit {result.returncode}): {result.stderr.strip()}"
            )
        return get_changed_files(repo)

    # --- lockfile delta -----------------------------------------------------------------

    def lockfile_delta(self, repo: Path) -> list[str]:
        """``name old->new`` pairs from Cargo.lock hunks (same block shape as uv.lock)."""
        try:
            diff_text = working_diff(repo)
        except VCSError:
            return []
        pairs = _parse_lockfile_diff(diff_text, _CARGO_LOCK_HEADER_RE)
        # Same-version pairs are metadata-only churn (checksum/source rewrites), not delta.
        return [p for p in pairs if not _is_same_version(p)]

    # --- audit --------------------------------------------------------------------------

    def run_audit(self, repo: Path, *, timeout: int | None = None) -> list[AuditFinding]:
        """cargo-audit primary, osv-scanner fallback; neither installed fails CLOSED."""
        timeout = timeout if timeout is not None else settings.audit_timeout
        try:
            return self._cargo_audit(repo, timeout)
        except FileNotFoundError:
            pass
        try:
            return self._osv_scanner(repo, timeout)
        except FileNotFoundError:
            raise AuditError(
                "neither cargo-audit nor osv-scanner is installed — cannot audit Cargo "
                "deps (fail-closed). Install one: `cargo install cargo-audit`"
            ) from None

    def _cargo_audit(self, repo: Path, timeout: int) -> list[AuditFinding]:
        proc = subprocess.run(
            ["cargo-audit", "audit", "--json"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        # Exit 1 = vulnerabilities found = a successful scan (live-verified).
        if proc.returncode not in (0, 1):
            raise AuditError(f"cargo-audit failed (exit {proc.returncode}): {proc.stderr.strip()}")
        return _parse_cargo_audit(proc.stdout)

    def _osv_scanner(self, repo: Path, timeout: int) -> list[AuditFinding]:
        proc = subprocess.run(
            ["osv-scanner", "--format", "json", "--lockfile", "Cargo.lock"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode not in (0, 1):
            raise AuditError(f"osv-scanner failed (exit {proc.returncode}): {proc.stderr.strip()}")
        from forge.dependabot.ecosystems.golang import _parse_osv_scanner

        return _parse_osv_scanner(proc.stdout)

    # --- evidence -----------------------------------------------------------------------

    def collect_evidence(
        self,
        candidate: BumpCandidate,
        findings: list[AuditFinding],
        lock_delta: list[str],
        *,
        repo_root: Path,
    ) -> EvidenceBundle:
        """Audit split + lockfile delta only; complete=False by construction (module doc)."""
        from forge.dependabot.supply_chain import split_findings

        findings_current, findings_target = split_findings(findings, candidate)
        return EvidenceBundle(
            candidate=candidate,
            findings_current=findings_current,
            findings_target=findings_target,
            lockfile_changes=lock_delta,
            complete=False,
            incomplete_reason=_CARGO_INCOMPLETE_REASON,
        )


# ---------------------------------------------------------------------------
# Manifest / lockfile / index helpers
# ---------------------------------------------------------------------------

_DEP_SECTIONS = ("dependencies", "dev-dependencies", "build-dependencies")


def _direct_deps(repo: Path) -> set[str]:
    """Registry-versioned direct crate names from the root Cargo.toml (renames resolved).

    Includes ``[workspace.dependencies]``; skips entries without a ``version`` (path/git
    deps have no registry latest to compare against)."""
    try:
        manifest = tomllib.loads((repo / "Cargo.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return set()
    tables: list[dict] = [manifest.get(s) or {} for s in _DEP_SECTIONS]
    tables.append((manifest.get("workspace") or {}).get("dependencies") or {})
    crates: set[str] = set()
    for table in tables:
        for alias, entry in table.items():
            if isinstance(entry, str):
                crates.add(alias)
            elif isinstance(entry, dict) and "version" in entry:
                crates.add(str(entry.get("package") or alias))
    return crates


def _locked_versions(repo: Path) -> dict[str, str]:
    """Locked version per registry-sourced crate; crates at multiple versions are DROPPED
    (an ambiguous current is not a candidate)."""
    try:
        lock = tomllib.loads((repo / "Cargo.lock").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    seen: dict[str, set[str]] = {}
    for pkg in lock.get("package") or []:
        if not str(pkg.get("source") or "").startswith("registry+"):
            continue
        name, version = pkg.get("name"), pkg.get("version")
        if name and version:
            seen.setdefault(str(name), set()).add(str(version))
    return {name: versions.pop() for name, versions in seen.items() if len(versions) == 1}


def _index_prefix(crate: str) -> str:
    if len(crate) == 1:
        return "1"
    if len(crate) == 2:
        return "2"
    if len(crate) == 3:
        return f"3/{crate[0]}"
    return f"{crate[:2]}/{crate[2:4]}"


def _index_latest(crate: str, *, timeout: float) -> str | None:
    """Max non-yanked, int-tuple-parseable version from the sparse index; None on ANY
    failure (a candidate is never fabricated from partial data)."""
    lowered = crate.lower()
    url = _INDEX_URL.format(prefix=_index_prefix(lowered), name=lowered)
    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
        lines = response.text.splitlines()
    except Exception:  # noqa: BLE001 — best-effort per candidate, fail toward "no candidate"
        return None
    best: tuple[tuple[int, ...], str] | None = None
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("yanked"):
            continue
        version = str(entry.get("vers") or "")
        parsed = _version_tuple(version)
        if parsed is None:  # prereleases and exotics — never "latest"
            continue
        if best is None or parsed > best[0]:
            best = (parsed, version)
    return best[1] if best else None


def _is_same_version(pair: str) -> bool:
    _, _, delta = pair.rpartition(" ")
    old, _, new = delta.partition("->")
    return old == new


def _parse_cargo_audit(stdout: str) -> list[AuditFinding]:
    """cargo-audit JSON → findings; RUSTSEC ids, patched-range lower bounds as fixes."""
    if not stdout.strip():
        return []
    try:
        data = json.loads(stdout)
    except ValueError as e:
        raise AuditError(f"cargo-audit --json output unparseable: {e}") from None
    merged: dict[tuple[str, str], AuditFinding] = {}
    for item in (data.get("vulnerabilities") or {}).get("list") or []:
        advisory = item.get("advisory") or {}
        package = str((item.get("package") or {}).get("name") or advisory.get("package") or "")
        vuln_id = str(advisory.get("id") or "")
        if not package or not vuln_id or (package, vuln_id) in merged:
            continue
        patched = (item.get("versions") or {}).get("patched") or []
        fixes = [m for rng in patched for m in _PATCHED_LOWER_RE.findall(str(rng))]
        merged[(package, vuln_id)] = AuditFinding(
            package=package,
            vuln_id=vuln_id,
            fix_versions=fixes,
            description=str(advisory.get("title") or "")[:300],
            aliases=[str(a) for a in (advisory.get("aliases") or [])],
        )
    return list(merged.values())
