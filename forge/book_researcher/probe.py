"""`forge book probe` — test which source hosts the tool proxy's egress can actually fetch.

A question that points at a blocked repository is unanswerable no matter how well it is phrased,
and the first real run burned a sprint on exactly that (nycourts.gov and gao.gov 403 the egress;
courtlistener.com and efts.sec.gov answer). Reachability is decidable without a model, so this
is the zero-token half of outline maintenance: probe every host the outline names, write the
verdicts to ``reachability.json`` in the project dir, and let the brief inject them into every
prompt. Re-run whenever the egress changes.

The fetcher is injectable so the verdict logic is unit-testable offline.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx

from forge.book_researcher.models import BookConfig, HostState, HostVerdict, Reachability

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Bare hostnames as they appear in prose ("courtlistener.com", "efts.sec.gov", "fec.gov/data").
# Deliberately narrow TLD list: a broad one turns "e.g." and "vs." into hosts.
_HOST_RE = re.compile(
    r"\b((?:[a-z0-9-]+\.)+(?:gov|org|com|net|edu|io|uk|eu|int|mil|info|us|ca|de|fr|ai))"
    r"(?:/[^\s,;)\]]*)?",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://[^\s)\]>,;]+", re.IGNORECASE)

# A probe status → state. Root-URL semantics, so a 404 at "/" still means the host answers and
# documents under it are fetchable (fec.gov/data 404s at the root of some mirrors).
_BLOCKED = frozenset({401, 403, 407, 451})
_UNKNOWN = frozenset({429, 500, 502, 503, 504})


def _host_of(text: str) -> str | None:
    """Normalise a URL or bare host+path fragment to its hostname."""
    text = text.strip().rstrip(".,;:")
    if "://" in text:
        return urlparse(text).hostname
    return text.split("/", 1)[0].lower() or None


def hosts_in_config(book: BookConfig) -> set[str]:
    """Every host the outline names: source policy, chapter sources, and hosts/URLs in prose."""
    hosts: set[str] = set()
    for item in [*book.sources.reachable, *book.sources.blocked]:
        h = _host_of(item)
        if h:
            hosts.add(h)
    texts: list[str] = [book.description, *book.sources.notes, *book.guidance]
    for ch in book.chapters:
        for item in ch.sources:
            h = _host_of(item)
            if h:
                hosts.add(h)
        texts += [ch.description, *ch.research_questions, *ch.guidance]
    for text in texts:
        for url in _URL_RE.findall(text):
            h = _host_of(url)
            if h:
                hosts.add(h)
        for m in _HOST_RE.finditer(text):
            hosts.add(m.group(1).lower())
    return hosts


def classify_status(status: int | None, error: str | None = None) -> tuple[HostState, str]:
    if status is None:
        return "unknown", f"unreachable ({error or 'no response'})"
    if status in _BLOCKED:
        return "blocked", f"HTTP {status}"
    if status in _UNKNOWN:
        return "unknown", f"HTTP {status} (transient — re-probe)"
    return "reachable", f"HTTP {status}"


Fetcher = Callable[[str], tuple[int | None, str | None]]
"""``fetch(url) -> (status, error)``: status None means no HTTP response at all."""


async def _fetch_all(
    urls: list[str], *, timeout: float, proxy: str | None
) -> list[tuple[int | None, str | None]]:
    async def one(client: httpx.AsyncClient, url: str) -> tuple[int | None, str | None]:
        try:
            # GET, not HEAD: plenty of sites answer HEAD with 405 or a misleading status.
            resp = await client.get(url)
            return resp.status_code, None
        except Exception as e:  # noqa: BLE001 — a probe reports, it never raises
            return None, type(e).__name__

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=timeout,
        headers={"User-Agent": _UA},
        limits=httpx.Limits(max_connections=8),
        proxy=proxy,
    ) as client:
        return list(await asyncio.gather(*(one(client, u) for u in urls)))


def _http_fetcher(timeout: float, proxy: str | None) -> Callable[[list[str]], list]:
    def fetch(urls: list[str]) -> list[tuple[int | None, str | None]]:
        try:
            return asyncio.run(_fetch_all(urls, timeout=timeout, proxy=proxy))
        except ImportError as e:  # socks proxy without httpx[socks]
            print(
                f"  NOTE: proxy unusable ({e}); probing directly — verdicts may not match egress."
            )
            return asyncio.run(_fetch_all(urls, timeout=timeout, proxy=None))

    return fetch


def probe_hosts(
    hosts: set[str],
    *,
    timeout: float = 15.0,
    proxy: str | None = None,
    fetch_many: Callable[[list[str]], list[tuple[int | None, str | None]]] | None = None,
) -> dict[str, HostVerdict]:
    """Fetch ``https://<host>/`` for each host and classify. Sorted, deterministic output."""
    ordered = sorted(hosts)
    urls = [f"https://{h}/" for h in ordered]
    fetch_many = fetch_many or _http_fetcher(timeout, proxy)
    results = fetch_many(urls) if urls else []
    verdicts: dict[str, HostVerdict] = {}
    for host, url, (status, error) in zip(ordered, urls, results, strict=True):
        state, note = classify_status(status, error)
        verdicts[host] = HostVerdict(host=host, url=url, status=status, state=state, note=note)
    return verdicts


def reachability_path(project_dir: Path) -> Path:
    return project_dir / "reachability.json"


def load_reachability(project_dir: Path) -> Reachability | None:
    path = reachability_path(project_dir)
    if not path.is_file():
        return None
    try:
        return Reachability.model_validate_json(path.read_text())
    except Exception:  # noqa: BLE001 — a corrupt probe file must not block a run
        print(f"  WARNING: could not parse {path}; ignoring it (re-run `forge book probe`).")
        return None


def save_reachability(project_dir: Path, reach: Reachability) -> Path:
    path = reachability_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(reach.model_dump_json(indent=2))
    return path


def probe_book(
    book: BookConfig,
    project_dir: Path,
    *,
    extra_hosts: list[str] | None = None,
    timeout: float = 15.0,
    proxy: str | None = None,
    fetch_many: Callable[[list[str]], list[tuple[int | None, str | None]]] | None = None,
) -> Reachability:
    """Probe every host the outline names (plus ``extra_hosts``), merge with the previous probe
    (a host that dropped out of the outline keeps its last verdict), and persist."""
    hosts = hosts_in_config(book)
    for extra in extra_hosts or []:
        host = _host_of(extra)
        if host:
            hosts.add(host)
    verdicts = probe_hosts(hosts, timeout=timeout, proxy=proxy, fetch_many=fetch_many)
    previous = load_reachability(project_dir)
    merged = dict(previous.hosts) if previous else {}
    merged.update(verdicts)
    reach = Reachability(
        probed_at=datetime.now(UTC).isoformat(timespec="seconds"),
        proxy=proxy,
        hosts=dict(sorted(merged.items())),
    )
    save_reachability(project_dir, reach)
    return reach


def render_reachability(reach: Reachability, *, only: set[str] | None = None) -> str:
    rows = [v for h, v in reach.hosts.items() if only is None or h in only]
    if not rows:
        return "(no hosts probed)"
    width = max(len(v.host) for v in rows)
    lines = [f"Probed {reach.probed_at}" + (f" via {reach.proxy}" if reach.proxy else " (direct)")]
    for v in rows:
        lines.append(f"  {v.state:<9} {v.host:<{width}}  {v.note}")
    counts = {s: sum(1 for v in rows if v.state == s) for s in ("reachable", "blocked", "unknown")}
    lines.append(
        f"  {counts['reachable']} reachable, {counts['blocked']} blocked, "
        f"{counts['unknown']} unknown"
    )
    return "\n".join(lines)
