"""The pnpm ecosystem adapter — JS/TS repos scanned, bumped, and audited via pnpm.

CLI surfaces pinned by live capture on 2026-07-12 (pnpm 11.8.0, fleet clones):

- ``pnpm outdated --json`` exits 1 when outdated packages exist (0 when current); stdout
  is a JSON object keyed by package name with latest/wanted/isDeprecated/dependencyType.
  There is NO "current" field on an uninstalled clone — the locked version comes from
  ``pnpm-lock.yaml`` (``importers`` → ``.`` → dependencies/devDependencies → name →
  version, peer-dep suffix like ``(react@18.2.0)`` stripped).
- ``pnpm update <name> --ignore-scripts`` updates the lock AND rewrites the package.json
  specifier (Dependabot-style range bump — both files are manifests, so the manifest-only
  gate is unaffected). ``--lockfile-only`` is silently IGNORED by update (live-verified):
  a real install lands in node_modules (gitignored, invisible to the VCS gates) — which
  is exactly why ``--ignore-scripts`` is NON-NEGOTIABLE: without it every dependency's
  lifecycle scripts execute on the host.
- ``pnpm audit --json`` works from the lockfile alone; exits 1 when advisories exist;
  npm-audit-v1 schema ``{advisories: {id: {...}}}``. ``patched_versions`` is a RANGE
  (e.g. ``>=7.11.1``) — fix versions are its ``>=`` lower bounds. ``github_advisory_id``
  (GHSA) is the durable vuln id; the numeric ``id`` is the fallback.
- Workspaces: v1 reads the ROOT importer (``.``) only — workspace-member deps are visible
  to the audit but not bumped (documented limitation; ``-r`` support is a follow-up).

Evidence posture: like Go, there is no npm-native provenance source wired yet, so
``collect_evidence`` returns ``complete=False`` with an explicit ``incomplete_reason`` —
every pnpm bump rides the advisory track by construction. The npm registry's deprecated/
time fields are the follow-up that would open the auto track.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import yaml

from forge.dependabot.audit import AuditError
from forge.dependabot.bump import BumpError
from forge.dependabot.config import settings
from forge.dependabot.models import AuditFinding, BumpCandidate, EvidenceBundle
from forge.dependabot.scan import _DELTA_ORDER, ScanError, classify_delta
from forge.dependabot.supply_chain import split_findings
from forge.shared.automerge import working_diff
from forge.task_worker.vcs import VCSError, get_changed_files, revert_changes

_PNPM_INCOMPLETE_REASON = (
    "no npm-native provenance evidence source yet (yanked/release-age/attestation/Scorecard "
    "signals are PyPI-only) — pnpm bumps ride the advisory track by construction"
)


class PnpmEcosystem:
    name = "pnpm"

    # --- scan ---------------------------------------------------------------------------

    def scan_outdated(self, repo: Path, *, timeout: int | None = None) -> list[BumpCandidate]:
        """Direct root-importer deps with a newer version, patch-first, capped.

        ``current`` is the LOCKED version from pnpm-lock.yaml; an outdated entry whose
        locked version cannot be found is dropped (an unprovable current is not a
        candidate), never guessed from ``wanted``.
        """
        timeout = timeout if timeout is not None else settings.scan_timeout
        result = subprocess.run(
            ["pnpm", "outdated", "--json"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        # Exit 1 with JSON = outdated packages found (live-verified); other codes fail.
        if result.returncode not in (0, 1):
            raise ScanError(
                f"pnpm outdated --json failed (exit {result.returncode}): {result.stderr.strip()}"
            )
        stdout = result.stdout.strip()
        if not stdout or stdout == "{}":
            return []
        try:
            outdated = json.loads(stdout)
        except ValueError as e:
            raise ScanError(f"pnpm outdated --json output unparseable: {e}") from None

        locked = _locked_versions(repo)
        candidates: list[BumpCandidate] = []
        for pkg_name, info in outdated.items():
            current = locked.get(pkg_name)
            latest = str(info.get("latest") or "")
            if not current or not latest:
                continue
            candidates.append(
                BumpCandidate(
                    name=pkg_name,
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
        """``pnpm update <name> --ignore-scripts``; revert + BumpError on failure.

        --ignore-scripts is the security floor, not an option (see module docstring)."""
        timeout = timeout if timeout is not None else settings.scan_timeout
        result = subprocess.run(
            ["pnpm", "update", candidate.name, "--ignore-scripts"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            revert_changes(repo)
            raise BumpError(
                f"pnpm update {candidate.name!r} failed (exit {result.returncode}): "
                f"{result.stderr.strip()}"
            )
        return get_changed_files(repo)

    # --- lockfile delta -----------------------------------------------------------------

    def lockfile_delta(self, repo: Path) -> list[str]:
        """``name old->new`` pairs from pnpm-lock.yaml package-section headers."""
        try:
            diff_text = working_diff(repo)
        except VCSError:
            return []
        return _parse_pnpm_lock_diff(diff_text)

    # --- audit --------------------------------------------------------------------------

    def run_audit(self, repo: Path, *, timeout: int | None = None) -> list[AuditFinding]:
        """pnpm audit primary, osv-scanner fallback; neither installed fails CLOSED."""
        timeout = timeout if timeout is not None else settings.audit_timeout
        try:
            return self._pnpm_audit(repo, timeout)
        except FileNotFoundError:
            pass
        try:
            return self._osv_scanner(repo, timeout)
        except FileNotFoundError:
            raise AuditError(
                "neither pnpm nor osv-scanner is installed — cannot audit pnpm deps (fail-closed)"
            ) from None

    def _pnpm_audit(self, repo: Path, timeout: int) -> list[AuditFinding]:
        proc = subprocess.run(
            ["pnpm", "audit", "--json"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        # Exit 1 = advisories found = a successful scan (live-verified).
        if proc.returncode not in (0, 1):
            raise AuditError(f"pnpm audit failed (exit {proc.returncode}): {proc.stderr.strip()}")
        return _parse_pnpm_audit(proc.stdout)

    def _osv_scanner(self, repo: Path, timeout: int) -> list[AuditFinding]:
        proc = subprocess.run(
            ["osv-scanner", "--format", "json", "--lockfile", "pnpm-lock.yaml"],
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
        findings_current, findings_target = split_findings(findings, candidate)
        return EvidenceBundle(
            candidate=candidate,
            findings_current=findings_current,
            findings_target=findings_target,
            lockfile_changes=lock_delta,
            complete=False,
            incomplete_reason=_PNPM_INCOMPLETE_REASON,
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# Strip a peer-dep suffix from a locked version: "5.0.0(typescript@5.9.3)" -> "5.0.0".
_PEER_SUFFIX_RE = re.compile(r"\(.*$")


def _locked_versions(repo: Path) -> dict[str, str]:
    """Direct-dependency locked versions from the ROOT importer of pnpm-lock.yaml.

    Lockfile v9 nests them under ``importers: -> ".":``; older non-workspace lockfiles
    keep dependencies at the top level — both shapes are read. Entries are either
    ``{specifier, version}`` dicts (v9) or plain version strings (older)."""
    lock_path = repo / "pnpm-lock.yaml"
    try:
        data = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    root = data.get("importers", {}).get(".", data) if isinstance(data, dict) else {}
    locked: dict[str, str] = {}
    for section in ("dependencies", "devDependencies", "optionalDependencies"):
        for pkg_name, entry in (root.get(section) or {}).items():
            version = entry.get("version", "") if isinstance(entry, dict) else str(entry)
            version = _PEER_SUFFIX_RE.sub("", str(version)).strip()
            if version:
                locked[pkg_name] = version
    return locked


# pnpm-lock.yaml hunk header in a git/jj unified diff.
_PNPM_LOCK_HEADER_RE = re.compile(r"^(?:---|\+\+\+) .*/?pnpm-lock\.yaml$")
_FILE_HEADER_RE = re.compile(r"^(?:diff --git |--- |\+\+\+ )")
# A packages-section header line: two-space indent, optional YAML quotes, name@version
# with an optional peer suffix — e.g. ``-  prettier@3.8.1:`` / ``+  '@types/node@20.1.2':``.
# The name may itself contain @ (scopes); the version starts after the LAST @.
_PKG_HEADER_RE = re.compile(
    r"^(?P<sign>[-+])  '?(?P<key>@?[^\s'@]+(?:/[^\s'@]+)?@[^\s':()]+)(?:\([^)]*\))*'?:\s*$"
)


def _parse_pnpm_lock_diff(diff_text: str) -> list[str]:
    """Extract ``name old->new`` pairs from pnpm-lock.yaml package-section headers.

    Covers the candidate AND transitive churn (uv-delta parity). Same-version pairs
    (resolution-only churn) are filtered."""
    in_lock = False
    entries: dict[str, dict[str, str | None]] = {}
    order: list[str] = []
    for line in diff_text.splitlines():
        if _PNPM_LOCK_HEADER_RE.match(line):
            in_lock = True
            continue
        if _FILE_HEADER_RE.match(line):
            in_lock = False
            continue
        if not in_lock:
            continue
        m = _PKG_HEADER_RE.match(line)
        if not m:
            continue
        pkg_name, _, version = m.group("key").rpartition("@")
        if not pkg_name or not version:
            continue
        if pkg_name not in entries:
            entries[pkg_name] = {"removed": None, "added": None}
            order.append(pkg_name)
        key = "removed" if m.group("sign") == "-" else "added"
        entries[pkg_name][key] = version

    return [
        f"{pkg} {entries[pkg]['removed']}->{entries[pkg]['added']}"
        for pkg in order
        if entries[pkg]["removed"]
        and entries[pkg]["added"]
        and entries[pkg]["removed"] != entries[pkg]["added"]
    ]


# The >= lower bounds of a patched range: ">=7.11.1" / ">=1.2.3 <2 || >=2.0.1" -> fixes.
_PATCHED_LOWER_RE = re.compile(r">=\s*([0-9][\w.-]*)")


def _parse_pnpm_audit(stdout: str) -> list[AuditFinding]:
    """npm-audit-v1 advisories → findings; GHSA id preferred, patched-range lower bounds
    as fix versions (an un-parseable or absent patched range yields no fix versions —
    conservative: the split then treats the finding as unresolved at any target)."""
    if not stdout.strip():
        return []
    try:
        data = json.loads(stdout)
    except ValueError as e:
        raise AuditError(f"pnpm audit --json output unparseable: {e}") from None
    merged: dict[tuple[str, str], AuditFinding] = {}
    for advisory in (data.get("advisories") or {}).values():
        module = str(advisory.get("module_name") or "")
        vuln_id = str(advisory.get("github_advisory_id") or advisory.get("id") or "")
        if not module or not vuln_id or (module, vuln_id) in merged:
            continue
        fixes = _PATCHED_LOWER_RE.findall(str(advisory.get("patched_versions") or ""))
        cves = advisory.get("cves") or []
        merged[(module, vuln_id)] = AuditFinding(
            package=module,
            vuln_id=vuln_id,
            fix_versions=fixes,
            description=str(advisory.get("title") or "")[:300],
            aliases=[str(c) for c in cves],
        )
    return list(merged.values())
