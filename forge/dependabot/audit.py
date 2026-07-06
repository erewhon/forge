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


def run_audit(repo: Path, *, timeout: int | None = None) -> list[AuditFinding]:
    """Run pip-audit on an ``uv export``-generated requirements file.

    1. Exports to a temp file via ``export_requirements``.
    2. Runs ``uvx pip-audit -r <tmp> --format json --disable-pip``.
    3. Parses JSON output: each dependency entry with vulns yields one
       ``AuditFinding`` per vulnerability.

    pip-audit exits 0 (no vulns) or 1 (vulns found) — both treated as success.
    Any other exit code raises ``AuditError``.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = Path(tmpdir) / "requirements.txt"
        export_requirements(repo, tmpfile)

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

        findings: list[AuditFinding] = []
        for entry in json.loads(proc.stdout):
            deps = entry.get("dependencies", [])
            vuln_info = entry.get("vuln", {})
            for dep in deps:
                vulns = (
                    vuln_info.get("all_vulns", [vuln_info])
                    if "all_vulns" in vuln_info
                    else [vuln_info]
                )
                for v in vulns:
                    finding = AuditFinding(
                        package=dep["package_name"],
                        vuln_id=v.get("id", ""),
                        fix_versions=v.get("fix_versions", []),
                        description=v.get("description", ""),
                        aliases=v.get("aliases", []),
                    )
                    findings.append(finding)

        return findings


def findings_for(findings: list[AuditFinding], package: str, version: str) -> list[AuditFinding]:
    """Filter *findings* to those whose package name matches *package* (PEP 503 normalized).

    Version filtering is handled elsewhere (evidence leaf); this function only
    does name matching.
    """
    normalized_target = _normalize_name(package)
    return [f for f in findings if _normalize_name(f.package) == normalized_target]
