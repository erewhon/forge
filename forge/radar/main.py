"""``forge radar`` — inspect the living AI Technology Radar and run the source scanners.

Usage::

    forge radar init                # stand up the "AI Radar" Nous notebook + blip database
    forge radar status              # quadrant × ring grid + counts + recent moves (local JSON)
    forge radar status --nous       # …read from the Nous blip database instead
    forge radar status --home PATH  # read the radar under PATH/.forge/radar/blips.json
    forge radar show "Qwen3-Coder"  # one blip's full detail and evidence trail
    forge radar scan                # run the source adapters, accumulate the candidate feed
    forge radar scan --source hackernews --dry-run   # one source, don't persist
    forge radar candidates          # show the accumulated feed

Read + scan surface. The weekly synthesis (a separate workstream) reads the candidate feed and is
the only thing that creates or moves blips.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from forge.radar.candidates import (
    DEFAULT_JSONL_PATH,
    CandidateFeed,
    CandidateStore,
    JsonlCandidateStore,
    provision_candidate_store,
)
from forge.radar.harvest import harvest
from forge.radar.models import RING_ORDER, Blip, Quadrant, Radar
from forge.radar.sources import default_adapters, radar_http_client
from forge.radar.store import (
    DEFAULT_JSON_PATH,
    RADAR_NOTEBOOK_NAME,
    JsonRadarStore,
    RadarStore,
    provision_radar_store,
)


def _nous_client():
    """A real Nous daemon client, or a clear error if the optional ``nous`` extra is missing. Import
    is deferred so the local-JSON commands never require the extra."""
    from forge.task_worker.config import settings
    from forge.task_worker.nous_client import require_nous

    require_nous()
    from nous_mcp.daemon_client import NousDaemonClient

    return NousDaemonClient(base_url=settings.daemon_url)


def _default_store(home: Path | None) -> RadarStore:
    base = home or Path.cwd()
    return JsonRadarStore(base / DEFAULT_JSON_PATH)


def _read_store(args: argparse.Namespace) -> RadarStore | None:
    """The store the read commands load from: the Nous blip database when ``--nous`` is set,
    otherwise the local JSON store. Returns ``None`` (with a printed hint) when ``--nous`` is asked
    for but the radar has not been provisioned yet."""
    if getattr(args, "nous", False):
        store = provision_radar_store(_nous_client(), create=False)
        if store is None:
            print(
                f'No "{RADAR_NOTEBOOK_NAME}" notebook / "Radar Blips" database yet — '
                "run `forge radar init` first.",
                file=sys.stderr,
            )
        return store
    return _default_store(args.home)


def _feed_store(args: argparse.Namespace) -> CandidateStore:
    """The candidate-feed store: the Nous "Radar Candidates" database with ``--nous`` (created if
    absent — this is a write path), else the local JSONL feed."""
    if getattr(args, "nous", False):
        return provision_candidate_store(_nous_client(), notebook_name=RADAR_NOTEBOOK_NAME)
    base = args.home or Path.cwd()
    return JsonlCandidateStore(base / DEFAULT_JSONL_PATH)


def _blip_slugs(args: argparse.Namespace) -> set[str]:
    """The slugs of existing blips, so the harvest skips candidates already promoted. Empty when the
    blip store has not been provisioned yet."""
    if getattr(args, "nous", False):
        store = provision_radar_store(_nous_client(), create=False)
        return set(store.load().by_slug()) if store else set()
    return set(_default_store(args.home).load().by_slug())


def _recent_moves(radar: Radar, limit: int = 5) -> list[Blip]:
    """Blips that have moved, most-recently-moved first."""
    moved = [b for b in radar.blips if b.last_moved]
    moved.sort(key=lambda b: b.last_moved or "", reverse=True)
    return moved[:limit]


def render_status(radar: Radar) -> str:
    """The quadrant × ring grid, a total, and the most recent moves — the at-a-glance view."""
    if not radar.blips:
        return "Radar is empty — no blips yet. Feed candidates via the scanners workstream."

    counts = radar.counts()
    ring_labels = [r.value for r in RING_ORDER]
    col_w = max(9, *(len(r) for r in ring_labels))
    quad_w = max(len(q.value) for q in Quadrant)

    header = " " * quad_w + "  " + "".join(f"{r:>{col_w}}" for r in ring_labels)
    lines = [header, " " * quad_w + "  " + "".join("-" * col_w for _ in ring_labels)]
    for quad in Quadrant:
        row = f"{quad.value:<{quad_w}}  "
        row += "".join(f"{counts[quad][r] or '.':>{col_w}}" for r in RING_ORDER)
        lines.append(row)

    lines.append("")
    lines.append(f"{len(radar.blips)} blips total")

    moves = _recent_moves(radar)
    if moves:
        lines.append("")
        lines.append("Recent moves:")
        for b in moves:
            arrow = f"{b.ring_last.value if b.ring_last else '?'} → {b.ring.value}"
            lines.append(f"  {b.last_moved}  {b.name}  ({arrow})")
    return "\n".join(lines)


def render_blip(blip: Blip) -> str:
    """One blip's full detail, including the accreted evidence trail."""
    lines = [
        f"{blip.name}",
        f"  quadrant:   {blip.quadrant.value}",
        f"  ring:       {blip.ring.value}"
        + (f"  (was {blip.ring_last.value})" if blip.ring_last else ""),
        f"  first seen: {blip.first_seen}",
        f"  last seen:  {blip.last_seen}",
        f"  last moved: {blip.last_moved or '(never)'}",
    ]
    if blip.rationale:
        lines.append(f"  rationale:  {blip.rationale}")
    if blip.action:
        lines.append(f"  action:     {blip.action}")
    if blip.links:
        lines.append("  links:")
        lines += [f"    - {link}" for link in blip.links]
    if blip.evidence:
        lines.append("  evidence:")
        lines += [
            f"    {e.date}  {e.note}" + (f"  [{e.source}]" if e.source else "")
            for e in blip.evidence
        ]
    return "\n".join(lines)


def render_feed(feed: CandidateFeed, limit: int = 20) -> str:
    """The candidate feed at a glance: per-source counts and the top entries by durability then
    popularity (how many scans have surfaced it, then its score)."""
    if not feed.entries:
        return "Candidate feed is empty — run `forge radar scan`."

    by_source: dict[str, int] = {}
    for e in feed.entries:
        by_source[e.source] = by_source.get(e.source, 0) + 1

    lines = [f"{len(feed.entries)} candidates in the feed"]
    lines.append("  " + "  ".join(f"{src}:{n}" for src, n in sorted(by_source.items())))
    lines.append("")

    top = sorted(feed.entries, key=lambda e: (e.times_seen, e.score or 0), reverse=True)[:limit]
    lines.append(f"Top {len(top)}:")
    for e in top:
        hint = e.quadrant_hint.value if e.quadrant_hint else "?"
        score = f", score {e.score:g}" if e.score is not None else ""
        lines.append(f"  [{hint:<19}] {e.title}  (x{e.times_seen}{score}, {e.source})")
    return "\n".join(lines)


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--home", type=Path, default=None, help="Repo root (default: cwd)")
    parser.add_argument(
        "--nous", action="store_true", help="Read from the Nous blip database, not local JSON."
    )


def _cmd_status(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="forge radar status", description="Quadrant × ring grid, counts, and recent moves."
    )
    _add_source_args(parser)
    args = parser.parse_args(argv)
    store = _read_store(args)
    if store is None:
        return 1
    print(render_status(store.load()))
    return 0


def _cmd_show(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="forge radar show", description="Full detail and evidence trail for one blip."
    )
    parser.add_argument("name", help="Blip name (matched case/punctuation-insensitively).")
    _add_source_args(parser)
    args = parser.parse_args(argv)
    store = _read_store(args)
    if store is None:
        return 1
    blip = store.load().get(args.name)
    if blip is None:
        print(f"No blip matching {args.name!r}.", file=sys.stderr)
        return 1
    print(render_blip(blip))
    return 0


def _cmd_scan(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="forge radar scan",
        description="Run the source adapters and accumulate the candidate feed.",
    )
    _add_source_args(parser)
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Only run this adapter (repeatable). Default: all.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Fetch + filter but don't persist.")
    args = parser.parse_args(argv)

    adapters = default_adapters()
    if args.source:
        wanted = {s.lower() for s in args.source}
        adapters = [a for a in adapters if a.name.lower() in wanted]
        if not adapters:
            print(f"No adapters match {sorted(wanted)}.", file=sys.stderr)
            return 1

    feed_store = _feed_store(args)
    feed = feed_store.load()
    blip_slugs = _blip_slugs(args)

    with radar_http_client() as client:
        report = harvest(feed, adapters, client, blip_slugs=blip_slugs, today=date.today())

    print(report.render())
    if args.dry_run:
        print("\n(dry run — feed not saved)")
        return 0
    feed_store.save(feed)
    return 0


def _cmd_candidates(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="forge radar candidates", description="Show the accumulated candidate feed."
    )
    _add_source_args(parser)
    parser.add_argument("--limit", type=int, default=20, help="How many top entries to show.")
    args = parser.parse_args(argv)
    print(render_feed(_feed_store(args).load(), limit=args.limit))
    return 0


def _cmd_init(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="forge radar init",
        description='Find-or-create the "AI Radar" Nous notebook and "Radar Blips" database.',
    )
    parser.parse_args(argv)
    store = provision_radar_store(_nous_client(), create=True)
    assert store is not None  # create=True never returns None
    print(f'"{RADAR_NOTEBOOK_NAME}" ready.')
    print(f"  notebook: {store.notebook_id}")
    print(f"  database: {store.db_id}")
    print("Read it with `forge radar status --nous`.")
    return 0


_COMMANDS = {
    "init": _cmd_init,
    "status": _cmd_status,
    "show": _cmd_show,
    "scan": _cmd_scan,
    "candidates": _cmd_candidates,
}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:]) if argv is None else list(argv)
    if args and args[0] in _COMMANDS:
        return _COMMANDS[args[0]](args[1:])

    parser = argparse.ArgumentParser(
        prog="forge radar",
        description="Inspect the living AI Technology Radar (quadrants × Adopt/Trial/Assess/Hold).",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("init", help='Stand up the "AI Radar" Nous notebook + blip database.')
    sub.add_parser("status", help="Quadrant × ring grid, counts, and recent moves.")
    sub.add_parser("show", help="Full detail and evidence trail for one blip.")
    sub.add_parser("scan", help="Run the source adapters, accumulate the candidate feed.")
    sub.add_parser("candidates", help="Show the accumulated candidate feed.")
    parser.parse_args(args)  # --help exits 0; unknown command exits 2
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
