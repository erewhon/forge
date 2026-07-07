"""Assemble the ``EvidenceBundle`` the risk policy and sign-off lens judge.

The load-bearing rule: ``complete=True`` ONLY when every evidence fetch succeeded — missing
evidence must read as risk (the policy fails closed to advisory), never as absence of risk.
An empty audit-findings list is NOT incompleteness; a failed PyPI fetch or an empty lockfile
delta for a changed lock is.

Pure signal helpers are separated from the two PyPI fetchers so tests inject canned metadata
and never touch the network.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import httpx

from agents.dependabot.audit import findings_for
from agents.dependabot.config import settings
from agents.dependabot.models import AuditFinding, BumpCandidate, EvidenceBundle

Fetcher = Callable[[str], dict | None]

_PYPI_VERSION_URL = "https://pypi.org/pypi/{name}/{version}/json"
_PYPI_PROJECT_URL = "https://pypi.org/pypi/{name}/json"

# Popular PyPI package names for the typosquat check: a candidate ONE edit away from any of
# these (but not itself a member) is suspect. Curated, not exhaustive — false positives cost
# only an advisory branch.
_TOP_PACKAGES = frozenset(
    {
        "requests",
        "urllib3",
        "numpy",
        "pandas",
        "scipy",
        "django",
        "flask",
        "fastapi",
        "pydantic",
        "httpx",
        "aiohttp",
        "rich",
        "click",
        "typer",
        "pytest",
        "setuptools",
        "wheel",
        "pip",
        "boto3",
        "botocore",
        "certifi",
        "idna",
        "charset-normalizer",
        "python-dateutil",
        "pyyaml",
        "six",
        "attrs",
        "packaging",
        "cryptography",
        "pillow",
        "sqlalchemy",
        "alembic",
        "celery",
        "redis",
        "psycopg2",
        "pymongo",
        "lxml",
        "beautifulsoup4",
        "selenium",
        "scrapy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "tensorflow",
        "torch",
        "transformers",
        "openai",
        "anthropic",
        "langchain",
        "jinja2",
        "markupsafe",
        "werkzeug",
        "gunicorn",
        "uvicorn",
        "starlette",
        "tornado",
        "twisted",
        "paramiko",
        "fabric",
        "ansible",
        "docker",
        "kubernetes",
        "pytz",
        "tzdata",
        "toml",
        "tomli",
        "ujson",
        "orjson",
        "msgpack",
        "protobuf",
        "grpcio",
        "websockets",
        "asyncpg",
        "aiofiles",
        "watchdog",
        "loguru",
        "structlog",
        "sentry-sdk",
        "prometheus-client",
        "opentelemetry-api",
        "typing-extensions",
        "mypy",
        "ruff",
        "black",
        "isort",
        "flake8",
        "pylint",
        "coverage",
        "tox",
        "nox",
        "pre-commit",
        "virtualenv",
        "pipenv",
        "poetry",
        "hatchling",
        "flit",
        "twine",
        "build",
        "cython",
        "numba",
        "polars",
        "duckdb",
    }
)


def fetch_pypi_version(name: str, version: str, *, timeout: float | None = None) -> dict | None:
    """Version-level PyPI JSON (yanked flag, upload times, project_urls). None on ANY failure."""
    try:
        resp = httpx.get(
            _PYPI_VERSION_URL.format(name=name, version=version),
            timeout=timeout if timeout is not None else settings.metadata_timeout,
            follow_redirects=True,
        )
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def fetch_pypi_project(name: str, *, timeout: float | None = None) -> dict | None:
    """Package-level PyPI JSON (canonical project_urls). None on ANY failure."""
    try:
        resp = httpx.get(
            _PYPI_PROJECT_URL.format(name=name),
            timeout=timeout if timeout is not None else settings.metadata_timeout,
            follow_redirects=True,
        )
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


# --- pure signal helpers --------------------------------------------------------------------


def _version_tuple(v: str) -> tuple[int, ...] | None:
    """Numeric version tuple (same spirit as scan.classify_delta); None if any part is junk."""
    try:
        return tuple(int(p) for p in v.lstrip("v").split("."))
    except ValueError:
        return None


def _fixed_at(target: str, fix_versions: list[str]) -> bool | None:
    """Is *target* at or past ANY fix version? None when unparseable (caller conservative)."""
    t = _version_tuple(target)
    if t is None or not fix_versions:
        return None
    verdicts = []
    for fv in fix_versions:
        f = _version_tuple(fv)
        if f is None:
            continue
        width = max(len(t), len(f))
        verdicts.append(t + (0,) * (width - len(t)) >= f + (0,) * (width - len(f)))
    return any(verdicts) if verdicts else None


def split_findings(
    findings: list[AuditFinding], candidate: BumpCandidate
) -> tuple[list[AuditFinding], list[AuditFinding]]:
    """Split the candidate's findings into (fixed by this bump, still present at target).

    Conservative: a finding whose fix versions can't be parsed or compared lands in
    ``findings_target`` — an unprovable fix is not a fix.
    """
    current: list[AuditFinding] = []
    target: list[AuditFinding] = []
    for f in findings_for(findings, candidate.name, candidate.current):
        (current if _fixed_at(candidate.latest, f.fix_versions) else target).append(f)
    return current, target


def package_age_days(version_meta: dict, *, now: datetime | None = None) -> int | None:
    """Days since the target release's earliest file upload; None when undeterminable."""
    times = []
    for f in version_meta.get("urls", []):
        raw = f.get("upload_time_iso_8601")
        if raw:
            try:
                times.append(datetime.fromisoformat(raw.replace("Z", "+00:00")))
            except ValueError:
                continue
    if not times:
        return None
    now = now if now is not None else datetime.now(UTC)
    return max(0, (now - min(times)).days)


_CHANGELOG_KEYS = ("changelog", "changes", "release notes", "releasenotes", "release-notes")
_FALLBACK_KEYS = ("repository", "source", "homepage")


def changelog_url(project_urls: dict | None) -> str | None:
    """The changelog-ish project URL, else a repo/homepage fallback, else None."""
    if not project_urls:
        return None
    lowered = {k.lower(): v for k, v in project_urls.items() if v}
    for key in _CHANGELOG_KEYS:
        if key in lowered:
            return lowered[key]
    for key in _FALLBACK_KEYS:
        if key in lowered:
            return lowered[key]
    return None


def _osa_distance(a: str, b: str, *, cap: int = 2) -> int:
    """Optimal-string-alignment (Damerau-Levenshtein) distance, early-exit past *cap*."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev2: list[int] = []
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and ca == b[j - 2] and cb == a[i - 2]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[len(b)]


def typosquat_suspect(name: str) -> str | None:
    """The popular package this name is ONE edit away from; None for exact members."""
    lowered = name.lower()
    if lowered in _TOP_PACKAGES:
        return None
    for popular in _TOP_PACKAGES:
        if _osa_distance(lowered, popular) == 1:
            return popular
    return None


# --- the bundle ------------------------------------------------------------------------------


def collect_evidence(
    candidate: BumpCandidate,
    findings: list[AuditFinding],
    lock_delta: list[str],
    *,
    fetch_version: Callable[..., dict | None] = fetch_pypi_version,
    fetch_project: Fetcher = fetch_pypi_project,
    now: datetime | None = None,
) -> EvidenceBundle:
    """Assemble the bundle. ``complete`` is True iff BOTH PyPI fetches returned data AND the
    lockfile delta is non-empty (a changed lock with an unparseable delta is unproven change).
    Audit findings may legitimately be empty — emptiness there is not incompleteness."""
    version_meta = fetch_version(candidate.name, candidate.latest)
    project_meta = fetch_project(candidate.name)

    findings_current, findings_target = split_findings(findings, candidate)
    info = (version_meta or {}).get("info", {})
    project_urls = info.get("project_urls") or (project_meta or {}).get("info", {}).get(
        "project_urls"
    )

    return EvidenceBundle(
        candidate=candidate,
        findings_current=findings_current,
        findings_target=findings_target,
        target_yanked=bool(info["yanked"]) if "yanked" in info else None,
        package_age_days=package_age_days(version_meta, now=now) if version_meta else None,
        changelog_url=changelog_url(project_urls),
        typosquat_suspect=typosquat_suspect(candidate.name),
        lockfile_changes=lock_delta,
        complete=version_meta is not None and project_meta is not None and bool(lock_delta),
    )
