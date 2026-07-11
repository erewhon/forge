"""Assemble the ``EvidenceBundle`` the risk policy and sign-off lens judge.

The load-bearing rule: ``complete=True`` ONLY when every evidence fetch succeeded — missing
evidence must read as risk (the policy fails closed to advisory), never as absence of risk.
An empty audit-findings list is NOT incompleteness; a failed PyPI fetch or an empty lockfile
delta for a changed lock is.

Pure signal helpers are separated from the two PyPI fetchers so tests inject canned metadata
and never touch the network.

Observed PyPI API shapes (verified 2026-07-11):
- Version JSON ``/pypi/{name}/{version}/json``: ``urls`` entries have ``filename``,
  ``packagetype`` (``bdist_wheel`` or ``sdist``), ``has_sig`` (GPG signature, NOT PEP 740),
  and other standard fields. No provenance marker in the version JSON itself.
- Integrity API ``/integrity/{name}/{version}/{filename}/provenance``: returns 200 with a
  JSON body containing ``attestation_bundles`` when a PEP 740 provenance attestation exists,
  and 404 when it does not. This is the sole signal for attestation presence.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx

from forge.dependabot.audit import findings_for
from forge.dependabot.config import settings
from forge.dependabot.models import AuditFinding, BumpCandidate, EvidenceBundle
from forge.dependabot.reachability import is_imported as _default_is_imported

Fetcher = Callable[[str], dict | None]

PYPI_INTEGRITY_PROVENANCE_URL = "https://pypi.org/integrity/{name}/{version}/{filename}/provenance"

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


_SCORECARD_URL = "https://api.securityscorecards.dev/projects/{host}/{owner}/{repo}"


def source_repo_url(project_urls: dict | None) -> str | None:
    """Extract a github.com/gitlab.com repository URL from PyPI project_urls.

    Scans keys like 'Source', 'Repository', and also a repo-shaped 'Homepage'.
    None when no recognizable repo URL is found.
    """
    if not project_urls:
        return None
    lowered = {k.lower(): v for k, v in project_urls.items() if v}
    # Direct repo keys first
    for key in ("source", "repository", "repo"):
        if key in lowered:
            return lowered[key]
    # Homepage that looks like a repo (github.com / gitlab.com)
    homepage = lowered.get("homepage")
    if homepage and ("github.com" in homepage or "gitlab.com" in homepage):
        return homepage
    return None


def fetch_scorecard(repo_url: str, *, timeout: float | None = None) -> dict | None:
    """GET the OpenSSF Scorecard for *repo_url*. None on any HTTP/parse failure."""
    try:
        # Expect URLs like https://github.com/owner/repo
        # API path: /projects/{host}/{owner}/{repo}
        cleaned = repo_url.rstrip("/")
        parts = cleaned.split("/")
        if len(parts) < 4:
            return None
        host = parts[2].replace(".com", "")
        owner = parts[3]
        repo = parts[4] if len(parts) > 4 else parts[-1]
        resp = httpx.get(
            _SCORECARD_URL.format(host=host, owner=owner, repo=repo),
            timeout=timeout if timeout is not None else settings.metadata_timeout,
            follow_redirects=True,
        )
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def scorecard_fields(
    data: dict | None,
) -> tuple[float | None, str | None]:
    """Extract (aggregate score, repo URL echo) from a Scorecard API payload.

    Returns (None, None) when data is None or the score is missing/unparseable.
    """
    if not data:
        return None, None
    try:
        score = data.get("scorecard", {}).get("score")
        if score is not None:
            score = float(score)
        repo = data.get("scorecard", {}).get("repo", {}).get("name")
        return score, repo
    except (ValueError, TypeError):
        return None, None


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


# --- attestation presence signal (PEP 740) --------------------------------------------------
# Best-effort by contract: None = "could not determine", which never marks evidence incomplete
# and never blocks on its own; only a provably-True signal is a policy stop.


def _wheel_or_sdist_filename(version_meta: dict | None) -> list[str] | None:
    """Return up to two filenames: first wheel, first sdist from the version release files.

    Returns None when there are no release files to inspect.
    """
    if not version_meta:
        return None
    filenames: list[str] = []
    seen_types: set[str] = set()
    for f in version_meta.get("urls", []):
        pt = f.get("packagetype", "")
        if pt in ("bdist_wheel", "sdist") and pt not in seen_types:
            fn = f.get("filename")
            if fn:
                filenames.append(fn)
                seen_types.add(pt)
        if len(seen_types) == 2:
            break
    return filenames if filenames else None


def fetch_attestation(
    name: str, version: str, filename: str, *, timeout: float | None = None
) -> bool | None:
    """Check whether *filename* of package *name* version *version* has a PEP 740 provenance
    attestation on PyPI.

    Returns True for 200 (attested), False for 404 (not attested), None on any other failure.
    """
    try:
        resp = httpx.get(
            PYPI_INTEGRITY_PROVENANCE_URL.format(name=name, version=version, filename=filename),
            timeout=timeout if timeout is not None else settings.metadata_timeout,
            follow_redirects=True,
        )
        if resp.status_code == 200:
            return True
        if resp.status_code == 404:
            return False
        return None
    except Exception:
        return None


def target_attestation(
    name: str,
    version_meta: dict | None,
    *,
    integrity_fetcher: Callable[[str, str, str], bool | None] | None = None,
) -> bool | None:
    """Determine whether the target version has any PEP 740 provenance attestation.

    Two sources, in order of preference:
    1. If an ``integrity_fetcher`` is injected (for tests), use it for the first sdist and
       first wheel filenames extracted from *version_meta*.
    2. Otherwise, hit the PyPI integrity API directly.

    Returns True if at least one release file is attested, False if lookups succeeded and
    none are, None on any fetch failure.
    """
    filenames = _wheel_or_sdist_filename(version_meta)
    if filenames is None or not filenames:
        # No release files to check — can't determine
        return None

    # Use the first sdist and first wheel (at most 2 requests)
    to_check = []
    for fn in filenames:
        if len(to_check) >= 2:
            break
        to_check.append(fn)

    if version_meta is None:
        return None
    fetcher = integrity_fetcher if integrity_fetcher is not None else fetch_attestation

    attested = False
    for fn in to_check:
        result = fetcher(name, version_meta.get("info", {}).get("version", "") or "", fn)
        if result is None:
            return None  # any failure -> undeterminable
        if result:
            attested = True
            break  # found at least one attested file — done

    return attested


# --- v2 provenance signals --------------------------------------------------------------------
# Best-effort by contract: None = "could not determine", which never marks evidence incomplete
# and never blocks on its own; only a provably-True signal is a policy stop.

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def _identity(info: dict) -> tuple[frozenset[str], str] | None:
    """(email set, name string) from a release's author/maintainer fields; None when the
    release exposes no identity at all. Emails are extracted from BOTH author_email and
    maintainer_email (projects move names between the fields; PyPI capture 2026-07-06 shows
    idna embedding the name in author_email while mcp splits author + maintainer_email)."""
    emails = frozenset(
        e.lower()
        for field in (info.get("author_email"), info.get("maintainer_email"))
        if field
        for e in _EMAIL_RE.findall(field)
    )
    names = " / ".join(
        str(info.get(k)).strip().lower() for k in ("author", "maintainer") if info.get(k)
    )
    if not emails and not names:
        return None
    return emails, names


def maintainer_change(current_info: dict | None, target_info: dict | None) -> bool | None:
    """Did the author/maintainer identity change between the two releases?

    Emails are the primary key (a set difference is a change); name strings only decide when
    NEITHER release exposes an email. None when either side exposes no identity to compare.
    """
    if current_info is None or target_info is None:
        return None
    cur, tgt = _identity(current_info), _identity(target_info)
    if cur is None or tgt is None:
        return None
    cur_emails, cur_names = cur
    tgt_emails, tgt_names = tgt
    if cur_emails or tgt_emails:
        return cur_emails != tgt_emails
    return cur_names != tgt_names


_SDIST_MAX_BYTES = 15 * 1024 * 1024  # far above the typical few-hundred-KB sdist
_MEMBER_MAX_BYTES = 256 * 1024


def _sdist_url(version_meta: dict | None) -> str | None:
    for f in (version_meta or {}).get("urls", []):
        if f.get("packagetype") == "sdist" and f.get("url"):
            return f["url"]
    return None


def fetch_sdist(
    url: str, *, timeout: float | None = None, max_bytes: int = _SDIST_MAX_BYTES
) -> bytes | None:
    """Download an sdist into memory, aborting past *max_bytes*. None on ANY failure."""
    try:
        with httpx.stream(
            "GET",
            url,
            timeout=timeout if timeout is not None else settings.metadata_timeout,
            follow_redirects=True,
        ) as resp:
            if resp.status_code != 200:
                return None
            chunks: list[bytes] = []
            total = 0
            for chunk in resp.iter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    return None
                chunks.append(chunk)
        return b"".join(chunks)
    except Exception:
        return None


def _sdist_traits(data: bytes) -> tuple[bool, str | None] | None:
    """(has top-level setup.py, build-backend or None) from sdist bytes, WITHOUT extracting to
    disk — member names plus a size-capped read of pyproject.toml only, nothing executed.
    Handles tar.* and zip sdists. None when the archive can't be read."""
    import io
    import tarfile
    import tomllib
    import zipfile

    def _traits(names: list[str], read: Callable[[str], bytes | None]) -> tuple[bool, str | None]:
        # sdist members live under one root dir: match depth <= 2 paths only.
        def top(name: str, leaf: str) -> bool:
            parts = name.split("/")
            return parts[-1] == leaf and len(parts) <= 2

        has_setup = any(top(n, "setup.py") for n in names)
        backend = None
        pyproject = next((n for n in names if top(n, "pyproject.toml")), None)
        if pyproject:
            raw = read(pyproject)
            if raw is not None:
                try:
                    backend = (
                        tomllib.loads(raw.decode("utf-8"))
                        .get("build-system", {})
                        .get("build-backend")
                    )
                except (tomllib.TOMLDecodeError, UnicodeDecodeError):
                    backend = None
        return has_setup, backend

    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:

            def read_tar(name: str) -> bytes | None:
                member = tf.getmember(name)
                if member.size > _MEMBER_MAX_BYTES:
                    return None
                fh = tf.extractfile(member)
                return fh.read() if fh else None

            return _traits(tf.getnames(), read_tar)
    except tarfile.TarError:
        pass
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:

            def read_zip(name: str) -> bytes | None:
                if zf.getinfo(name).file_size > _MEMBER_MAX_BYTES:
                    return None
                return zf.read(name)

            return _traits(zf.namelist(), read_zip)
    except (zipfile.BadZipFile, KeyError):
        return None


def install_script_change(
    current_meta: dict | None,
    target_meta: dict | None,
    *,
    fetch_bytes: Callable[[str], bytes | None] = fetch_sdist,
) -> bool | None:
    """Does the TARGET release gain install/build-script surface over the current one?

    True when the target sdist introduces a top-level setup.py the current lacks, or when both
    declare build backends and they differ. False when neither release ships an sdist (a pure
    wheel has no install-time script surface) or when traits match. None whenever the
    comparison can't be made — a missing metadata fetch, a one-sided sdist, or a fetch/archive
    failure. Missing metadata is never proof of absence.
    """
    if current_meta is None or target_meta is None:
        return None
    cur_url, tgt_url = _sdist_url(current_meta), _sdist_url(target_meta)
    if cur_url is None and tgt_url is None:
        return False
    if cur_url is None or tgt_url is None:
        return None
    cur_data, tgt_data = fetch_bytes(cur_url), fetch_bytes(tgt_url)
    if cur_data is None or tgt_data is None:
        return None
    cur_traits, tgt_traits = _sdist_traits(cur_data), _sdist_traits(tgt_data)
    if cur_traits is None or tgt_traits is None:
        return None
    cur_setup, cur_backend = cur_traits
    tgt_setup, tgt_backend = tgt_traits
    if tgt_setup and not cur_setup:
        return True
    if cur_backend and tgt_backend and cur_backend != tgt_backend:
        return True
    return False


# --- the bundle ------------------------------------------------------------------------------


def collect_evidence(
    candidate: BumpCandidate,
    findings: list[AuditFinding],
    lock_delta: list[str],
    *,
    fetch_version: Callable[..., dict | None] = fetch_pypi_version,
    fetch_project: Fetcher = fetch_pypi_project,
    fetch_bytes: Callable[[str], bytes | None] = fetch_sdist,
    fetch_scorecard: Callable[[str], dict | None] = fetch_scorecard,
    fetch_attestation: Callable[[str, str, str], bool | None] = fetch_attestation,
    now: datetime | None = None,
    reachability_checker: Callable[[Path, str], bool | None] | None = None,
    repo_root: Path | None = None,
) -> EvidenceBundle:
    """Assemble the bundle. ``complete`` is True iff BOTH target-version PyPI fetches returned
    data AND the lockfile delta is non-empty (a changed lock with an unparseable delta is
    unproven change). Audit findings may legitimately be empty — emptiness there is not
    incompleteness. The v2 provenance signals (maintainer change, install scripts) compare the
    CURRENT release's metadata/sdist against the target's; their unavailability yields None and
    deliberately does not affect ``complete`` — they gate only when provably True. The Scorecard
    signal (fetch_scorecard) is also best-effort; unavailability never affects ``complete``.
    The PEP 740 attestation signal (fetch_attestation) is also best-effort; unavailability
    (None) never affects ``complete``."""
    version_meta = fetch_version(candidate.name, candidate.latest)
    project_meta = fetch_project(candidate.name)
    current_meta = fetch_version(candidate.name, candidate.current)

    findings_current, findings_target = split_findings(findings, candidate)
    info = (version_meta or {}).get("info", {})
    project_urls = info.get("project_urls") or (project_meta or {}).get("info", {}).get(
        "project_urls"
    )

    # Scorecard signal — best effort; the parameter defaults to the live fetcher (bound at
    # def time, same pattern as fetch_version) so production call sites that pass nothing
    # still fetch. Tests inject a lambda; simulate failure with `lambda u: None`.
    repo_url = source_repo_url(project_urls)
    sc_data = fetch_scorecard(repo_url) if repo_url else None
    sc_score, sc_repo = scorecard_fields(sc_data)

    # PEP 740 attestation signal — best effort, same live-fetcher default.
    att = (
        target_attestation(candidate.name, version_meta, integrity_fetcher=fetch_attestation)
        if version_meta
        else None
    )

    # Reachability signal — demote-only, injectable for tests.
    checker = reachability_checker if reachability_checker is not None else _default_is_imported
    reachable: bool | None = None
    if checker is not None and repo_root is not None:
        reachable = checker(repo_root, candidate.name)

    return EvidenceBundle(
        candidate=candidate,
        findings_current=findings_current,
        findings_target=findings_target,
        target_yanked=bool(info["yanked"]) if "yanked" in info else None,
        package_age_days=package_age_days(version_meta, now=now) if version_meta else None,
        changelog_url=changelog_url(project_urls),
        typosquat_suspect=typosquat_suspect(candidate.name),
        maintainer_changed=maintainer_change(
            (current_meta or {}).get("info") if current_meta else None,
            info if version_meta else None,
        ),
        new_install_scripts=install_script_change(
            current_meta, version_meta, fetch_bytes=fetch_bytes
        ),
        scorecard_score=sc_score,
        scorecard_repo=sc_repo,
        target_attested=att,
        reachable=reachable,
        lockfile_changes=lock_delta,
        complete=version_meta is not None and project_meta is not None and bool(lock_delta),
    )
