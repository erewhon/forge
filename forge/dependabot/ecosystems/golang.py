"""The Go ecosystem adapter — go modules scanned, bumped, and audited via the go toolchain.

CLI surfaces pinned by live capture on 2026-07-11 (go 1.26.4, govulncheck v1.6.0):

- ``go list -u -m -json all`` emits a stream of concatenated JSON module objects. The main
  module has ``"Main": true`` and no ``Version``; indirect deps carry ``"Indirect": true``;
  an available upgrade appears as ``"Update": {"Version": "vX.Y.Z"}``.
- ``govulncheck -scan package -json ./...`` is the audit invocation. Module scan
  (``-scan module``) looks like the obvious pip-audit analogue but is a trap twice over:
  it rejects patterns ("patterns are not accepted for module only scanning") AND its
  results depend on which package directory it runs from (18 findings from ``cmd/`` vs 27
  from ``cmd/soft/`` on the live capture) — a silent under-report. Package scan over
  ``./...`` loads the whole tree from the repo root and returned the identical 27
  module-vuln pairs as the best-case module scan. In ``-json`` mode it exits 0 even when
  vulnerabilities are found.
- The JSON stream's ``osv`` messages are the database CATALOG consulted (264 entries on the
  live capture), NOT results; the actual results are the ``finding`` messages (27 on the
  same capture). Findings carry the module, current version, and ``fixed_version``; the
  catalog is joined by id only for summary/aliases.

Evidence posture: there is no Go-native provenance source wired yet (yanked / release age /
attestation / Scorecard are PyPI-only), so ``collect_evidence`` always returns
``complete=False`` with an explicit ``incomplete_reason`` — every Go bump rides the
advisory track by construction. deps.dev is the candidate source for a future Go evidence
pass; until then the demotion is deliberate, not a gap.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from forge.dependabot.audit import AuditError
from forge.dependabot.bump import BumpError
from forge.dependabot.config import settings
from forge.dependabot.models import AuditFinding, BumpCandidate, EvidenceBundle
from forge.dependabot.scan import _DELTA_ORDER, ScanError, classify_delta
from forge.dependabot.supply_chain import split_findings
from forge.shared.automerge import working_diff
from forge.task_worker.vcs import VCSError, get_changed_files, revert_changes

_GO_INCOMPLETE_REASON = (
    "no Go-native provenance evidence source yet (yanked/release-age/attestation/Scorecard "
    "signals are PyPI-only) — Go bumps ride the advisory track by construction"
)


def _iter_json_objects(text: str):
    """Yield each object from a stream of concatenated JSON values (go tooling's format)."""
    decoder = json.JSONDecoder()
    idx, end = 0, len(text)
    while idx < end:
        while idx < end and text[idx] in " \t\r\n":
            idx += 1
        if idx >= end:
            break
        try:
            obj, idx = decoder.raw_decode(text, idx)
        except ValueError:
            break  # trailing non-JSON output — everything parseable was consumed
        yield obj


class GoEcosystem:
    name = "go"

    # --- scan ---------------------------------------------------------------------------

    def scan_outdated(self, repo: Path, *, timeout: int | None = None) -> list[BumpCandidate]:
        """Direct modules with an available update, patch-first, capped like the uv scan."""
        timeout = timeout if timeout is not None else settings.scan_timeout
        result = subprocess.run(
            ["go", "list", "-u", "-m", "-json", "all"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise ScanError(
                f"go list -u -m -json all failed (exit {result.returncode}): "
                f"{result.stderr.strip()}"
            )

        candidates: list[BumpCandidate] = []
        for mod in _iter_json_objects(result.stdout):
            if mod.get("Main") or mod.get("Indirect"):
                continue
            update = mod.get("Update")
            if not update or not update.get("Version") or not mod.get("Version"):
                continue
            current = str(mod["Version"]).lstrip("v")
            latest = str(update["Version"]).lstrip("v")
            candidates.append(
                BumpCandidate(
                    name=mod["Path"],
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
        """``go get <module>@v<latest>`` then ``go mod tidy``; revert + BumpError on failure."""
        timeout = timeout if timeout is not None else settings.scan_timeout
        commands = (
            ["go", "get", f"{candidate.name}@v{candidate.latest}"],
            ["go", "mod", "tidy"],
        )
        for cmd in commands:
            result = subprocess.run(
                cmd, cwd=str(repo), capture_output=True, text=True, timeout=timeout
            )
            if result.returncode != 0:
                revert_changes(repo)
                raise BumpError(
                    f"{' '.join(cmd)} failed (exit {result.returncode}): {result.stderr.strip()}"
                )
        return get_changed_files(repo)

    # --- lockfile delta -----------------------------------------------------------------

    def lockfile_delta(self, repo: Path) -> list[str]:
        """``module old->new`` pairs parsed from the working diff's go.mod hunks."""
        try:
            diff_text = working_diff(repo)
        except VCSError:
            return []
        return _parse_gomod_diff(diff_text)

    # --- audit --------------------------------------------------------------------------

    def run_audit(self, repo: Path, *, timeout: int | None = None) -> list[AuditFinding]:
        """govulncheck package scan, osv-scanner fallback; neither installed fails CLOSED."""
        timeout = timeout if timeout is not None else settings.audit_timeout
        try:
            return self._govulncheck(repo, timeout)
        except FileNotFoundError:
            pass
        try:
            return self._osv_scanner(repo, timeout)
        except FileNotFoundError:
            raise AuditError(
                "neither govulncheck nor osv-scanner is installed — cannot audit Go modules "
                "(fail-closed). Install one: `go install golang.org/x/vuln/cmd/govulncheck@latest`"
            ) from None

    def _govulncheck(self, repo: Path, timeout: int) -> list[AuditFinding]:
        # Package scan over ./... — NOT module scan, whose results silently depend on the
        # launch directory (see module docstring). A repo with no .go files at all fails
        # the package load with a nonzero exit, which is the fail-closed path we want.
        proc = subprocess.run(
            ["govulncheck", "-scan", "package", "-json", "./..."],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        # -json mode exits 0 even with findings (live-verified); 3 is the documented
        # vulns-found code kept for tolerance across versions.
        if proc.returncode not in (0, 3):
            raise AuditError(f"govulncheck failed (exit {proc.returncode}): {proc.stderr.strip()}")
        return _parse_govulncheck(proc.stdout)

    def _osv_scanner(self, repo: Path, timeout: int) -> list[AuditFinding]:
        proc = subprocess.run(
            ["osv-scanner", "--format", "json", "--lockfile", "go.sum"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        # osv-scanner exits 1 when vulnerabilities are found — that is a successful scan.
        if proc.returncode not in (0, 1):
            raise AuditError(f"osv-scanner failed (exit {proc.returncode}): {proc.stderr.strip()}")
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
        """Audit split + lockfile delta only; complete=False by construction (see module doc)."""
        findings_current, findings_target = split_findings(findings, candidate)
        return EvidenceBundle(
            candidate=candidate,
            findings_current=findings_current,
            findings_target=findings_target,
            lockfile_changes=lock_delta,
            complete=False,
            incomplete_reason=_GO_INCOMPLETE_REASON,
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# go.mod hunk header in a git/jj unified diff.
_GOMOD_HEADER_RE = re.compile(r"^(?:---|\+\+\+) .*/?go\.mod$")
# Any file header — used to LEAVE the go.mod section (same guard as bump.py's uv.lock parse).
_FILE_HEADER_RE = re.compile(r"^(?:diff --git |--- |\+\+\+ )")
# A changed requirement line: optional `require` keyword (single-line form), module path,
# a v-prefixed version, optional `// indirect` comment. `go`/`toolchain` directive lines
# never match because their versions lack the `v` prefix.
_REQ_RE = re.compile(
    r"^(?P<sign>[-+])\s*(?:require\s+)?(?P<path>[^\s()]+)\s+v(?P<ver>[0-9]\S*?)(?:\s+//.*)?$"
)


def _parse_gomod_diff(diff_text: str) -> list[str]:
    """Extract ``module old->new`` pairs from the go.mod hunks of a unified diff."""
    in_gomod = False
    entries: dict[str, dict[str, str | None]] = {}
    order: list[str] = []
    for line in diff_text.splitlines():
        if _GOMOD_HEADER_RE.match(line):
            in_gomod = True
            continue
        if _FILE_HEADER_RE.match(line):
            in_gomod = False
            continue
        if not in_gomod:
            continue
        m = _REQ_RE.match(line)
        if not m:
            continue
        path = m.group("path")
        if path not in entries:
            entries[path] = {"removed": None, "added": None}
            order.append(path)
        key = "removed" if m.group("sign") == "-" else "added"
        entries[path][key] = m.group("ver")

    # Same-version pairs are annotation churn (e.g. `// indirect` dropped when go mod tidy
    # promotes a stale annotation to a direct require) — not a version change, so not delta.
    return [
        f"{path} {entries[path]['removed']}->{entries[path]['added']}"
        for path in order
        if entries[path]["removed"]
        and entries[path]["added"]
        and entries[path]["removed"] != entries[path]["added"]
    ]


def _parse_govulncheck(stdout: str) -> list[AuditFinding]:
    """Join ``finding`` messages (the results) with the ``osv`` catalog (summary/aliases).

    The catalog lists every database entry whose module appears ANYWHERE in the dependency
    graph regardless of version — treating it as results would report hundreds of phantom
    vulnerabilities (live capture: 264 catalog entries vs 27 actual findings).
    """
    catalog: dict[str, dict] = {}
    raw_findings: list[dict] = []
    for obj in _iter_json_objects(stdout):
        if "osv" in obj and isinstance(obj["osv"], dict):
            catalog[obj["osv"].get("id", "")] = obj["osv"]
        elif "finding" in obj:
            raw_findings.append(obj["finding"])

    merged: dict[tuple[str, str], AuditFinding] = {}
    for f in raw_findings:
        trace = f.get("trace") or [{}]
        module = trace[0].get("module", "")
        vuln_id = f.get("osv", "")
        if not module or not vuln_id:
            continue
        fixed = str(f.get("fixed_version", "")).lstrip("v")
        key = (module, vuln_id)
        if key in merged:
            if fixed and fixed not in merged[key].fix_versions:
                merged[key].fix_versions.append(fixed)
            continue
        entry = catalog.get(vuln_id, {})
        merged[key] = AuditFinding(
            package=module,
            vuln_id=vuln_id,
            fix_versions=[fixed] if fixed else [],
            description=entry.get("summary", "") or (entry.get("details") or "")[:300],
            aliases=list(entry.get("aliases") or []),
        )
    return list(merged.values())


def _parse_osv_scanner(stdout: str) -> list[AuditFinding]:
    """Parse ``osv-scanner --format json`` results into findings (one per package+vuln)."""
    if not stdout.strip():
        return []
    data = json.loads(stdout)
    merged: dict[tuple[str, str], AuditFinding] = {}
    for result in data.get("results", []):
        for pkg in result.get("packages", []):
            name = pkg.get("package", {}).get("name", "")
            for vuln in pkg.get("vulnerabilities", []):
                vuln_id = vuln.get("id", "")
                if not name or not vuln_id or (name, vuln_id) in merged:
                    continue
                fixes = [
                    str(event["fixed"]).lstrip("v")
                    for aff in vuln.get("affected", [])
                    if aff.get("package", {}).get("name") == name
                    for rng in aff.get("ranges", [])
                    for event in rng.get("events", [])
                    if "fixed" in event
                ]
                merged[(name, vuln_id)] = AuditFinding(
                    package=name,
                    vuln_id=vuln_id,
                    fix_versions=fixes,
                    description=vuln.get("summary", "") or (vuln.get("details") or "")[:300],
                    aliases=list(vuln.get("aliases") or []),
                )
    return list(merged.values())
