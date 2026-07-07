from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

from agents.dependabot.models import AuditFinding


class AuditError(Exception):
    """Raised when the audit pipeline (export or pip-audit) fails."""


def _normalize_name(name: str) -> str:
    """Normalize a package name per PEP 503: lowercase, runs of [-_.] collapse to -."""
    return re.sub(r"[-_.]+", "-", name).lower()


def export_requirements(repo: Path, out: Path) -> None:
    """Export the uv lockfile to a requirements.txt via ``uv export``.

    Runs ``uv export --frozen --no-emit-project -o <out>`` and raises
    ``AuditError`` on any non-zero exit code (carrying stderr in the message).
    """
    result = subprocess.run(
        ["uv", "export", "--frozen", "--no-emit-project", "-o", str(out)],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AuditError(f"uv export failed (exit {result.returncode}): {result.stderr.strip()}")


def _filter_auditable(text: str) -> str:
    """Drop requirement blocks pip-audit cannot audit (path/editable/URL deps without pins).

    ``uv export`` emits local path deps (e.g. ``../nous/nous-py``) even with ``--no-emit-project``;
    they have no hash, which breaks pip-audit's hash-checking mode, and they aren't on PyPI to
    audit anyway. Keep comments/blanks and any requirement block whose header line contains
    ``==`` (pinned, hashable); a block's continuation lines (``--hash=...``) follow their
    header's fate. Verified against this repo's real export on 2026-07-06.
    """
    out: list[str] = []
    keep = True
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            out.append(line)
            continue
        if not line[:1].isspace():  # start of a new requirement block
            keep = "==" in line
        if keep:
            out.append(line)
    return "\n".join(out) + "\n"


def _parse_findings(stdout: str) -> list[AuditFinding]:
    """Parse ``pip-audit --format json`` output: a top-level object whose ``dependencies`` list
    holds ``{"name", "version", "vulns": [...]}`` entries (schema captured from a real run)."""
    data = json.loads(stdout)
    findings: list[AuditFinding] = []
    for dep in data.get("dependencies", []):
        for v in dep.get("vulns", []):
            findings.append(
                AuditFinding(
                    package=dep.get("name", ""),
                    vuln_id=v.get("id", ""),
                    fix_versions=v.get("fix_versions", []),
                    description=v.get("description", ""),
                    aliases=v.get("aliases", []),
                )
            )
    return findings


def run_audit(repo: Path, *, timeout: int | None = None) -> list[AuditFinding]:
    """Run pip-audit on an ``uv export``-generated requirements file.

    1. Exports to a temp file via ``export_requirements``.
    2. Filters out non-auditable entries (local path/editable deps) — see ``_filter_auditable``.
    3. Runs ``uvx pip-audit -r <tmp> --format json --disable-pip`` and parses the JSON to one
       ``AuditFinding`` per vulnerability per dependency.

    pip-audit exits 0 (no vulns) or 1 (vulns found) — both treated as success.
    Any other exit code raises ``AuditError``.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = Path(tmpdir) / "requirements.txt"
        export_requirements(repo, tmpfile)
        tmpfile.write_text(_filter_auditable(tmpfile.read_text()))

        proc = subprocess.run(
            ["uvx", "pip-audit", "-r", str(tmpfile), "--format", "json", "--disable-pip"],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if proc.returncode not in (0, 1):
            raise AuditError(f"pip-audit failed (exit {proc.returncode}): {proc.stderr.strip()}")

        if not proc.stdout.strip():
            return []
        return _parse_findings(proc.stdout)


def findings_for(findings: list[AuditFinding], package: str, version: str) -> list[AuditFinding]:
    """Filter *findings* to those whose package name matches *package* (PEP 503 normalized).

    Version filtering is handled elsewhere (evidence leaf); this function only
    does name matching.
    """
    normalized_target = _normalize_name(package)
    return [f for f in findings if _normalize_name(f.package) == normalized_target]
