"""``forge map`` — mirror checkouts and emit structural repo maps.

Usage::

    forge map targets                     # list configured targets and their mirror state
    forge map sync [--target NAME ...]    # rsync checkouts into <output_root>/mirror/
    forge map structure [--target NAME ...]   # tree-sitter pass → out/<key>/{map.md,index.json}
    forge map summarize [--target NAME ...]   # LLM stage (local router only; guarded) → summaries
    forge map render [--target NAME ...]      # assemble modules/*.md + architecture.md from cache
    forge map run [--target NAME ...]         # full pipeline: sync → structure → summarize → render

Config lives at ``~/.config/forge/map.toml`` (override with ``--config``)::

    output_root = "/home/me/contract-maps"     # keep on the home network — derived artifacts
    [[targets]]
    name = "food-api"
    host = "work-vm.m.example"                  # omit for a local path
    path = "/home/me/src/food-api"
    ignore_globs = ["generated"]

Exit code is 0 while at least one selected target succeeds; per-target failures degrade to
warnings (an unreachable VM must never sink the other targets).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from forge.cartographer.models import MapConfig, TargetConfig, TargetInfo, load_config
from forge.cartographer.structural import build_index, write_outputs
from forge.cartographer.sync import read_sync_state, sync_target


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="forge map", description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=None, help="path to map.toml")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, summary in (
        ("targets", "list configured targets and their mirror state"),
        ("sync", "mirror the configured checkouts"),
        ("structure", "run the structural pass over existing mirrors"),
        ("summarize", "run the guarded LLM stage over existing structural indexes"),
        ("render", "assemble module and architecture docs from the summary cache"),
        ("run", "full pipeline: sync, structure, summarize, render + report"),
    ):
        cmd = sub.add_parser(name, help=summary)
        if name != "targets":
            cmd.add_argument(
                "--target", action="append", dest="targets", metavar="NAME",
                help="limit to this target (repeatable; default: all)",
            )
    return parser.parse_args(argv)


def _load(args: argparse.Namespace) -> MapConfig:
    try:
        return load_config(args.config)
    except FileNotFoundError as exc:
        hint = f"forge map: no config at {exc.filename} — see `forge map --help`"
        raise SystemExit(hint) from exc


def _selected(config: MapConfig, args: argparse.Namespace) -> list[TargetConfig]:
    try:
        return config.target_selection(getattr(args, "targets", None))
    except KeyError as exc:
        raise SystemExit(f"forge map: {exc.args[0]}") from exc


def _cmd_targets(config: MapConfig) -> int:
    for target in config.targets:
        synced_at, stale = read_sync_state(config, target)
        state = "stale" if stale else f"synced {synced_at:%Y-%m-%d %H:%M}"
        origin = f"{target.host}:{target.path}" if target.host else target.path
        print(f"{target.name:24} {origin:56} [{state}]  key={target.key}")
    return 0


def _cmd_sync(config: MapConfig, targets: list[TargetConfig]) -> int:
    failures = 0
    for target in targets:
        result = sync_target(config, target)
        if result.ok:
            print(f"synced {target.name} → {config.mirror_dir(target)}")
        else:
            failures += 1
            print(f"WARN: sync failed for {target.name} (mirror kept, marked stale): "
                  f"{result.message}", file=sys.stderr)
    return 1 if targets and failures == len(targets) else 0


def _cmd_structure(config: MapConfig, targets: list[TargetConfig]) -> int:
    failures = 0
    for target in targets:
        synced_at, stale = read_sync_state(config, target)
        info = TargetInfo(
            name=target.name, host=target.host, path=target.path,
            synced_at=synced_at, stale=stale,
        )
        try:
            result = build_index(config, target, info)
        except (FileNotFoundError, RuntimeError) as exc:
            failures += 1
            print(f"WARN: structure failed for {target.name}: {exc}", file=sys.stderr)
            continue
        out = write_outputs(config, target, result)
        stale_note = " (STALE mirror)" if stale else ""
        print(f"mapped {target.name}{stale_note}: {result.parsed} files parsed, "
              f"{len(result.skipped)} skipped → {out}")
    return 1 if targets and failures == len(targets) else 0


def _cmd_summarize(config: MapConfig, targets: list[TargetConfig]) -> int:
    # Deferred import: plain sync/structure must never require the LLM settings stack.
    from forge.cartographer.config import settings
    from forge.cartographer.guards import run_guards
    from forge.cartographer.summarize import summarize_target

    run_guards(settings.openai_base_url, config.output_root)  # refuses via SystemExit
    failures = 0
    for target in targets:
        try:
            stats = summarize_target(config, settings, target)
        except FileNotFoundError as exc:
            failures += 1
            print(f"WARN: summarize failed for {target.name}: {exc}", file=sys.stderr)
            continue
        note = f", {len(stats.failed)} FAILED" if stats.failed else ""
        print(
            f"summarized {target.name}: {stats.summarized} new, {stats.cache_hits} cached, "
            f"{stats.rollups_built} rollups, {stats.llm_calls} LLM calls, "
            f"{len(stats.skipped)} skipped{note}"
        )
        for line in stats.skipped:
            print(f"  skipped {line}", file=sys.stderr)
        for line in dict.fromkeys(stats.errors):  # dedup, keep order
            print(f"  error {line}", file=sys.stderr)
    return 1 if targets and failures == len(targets) else 0


def _cmd_render(config: MapConfig, targets: list[TargetConfig]) -> int:
    from forge.cartographer.guards import guard_output_root
    from forge.cartographer.render import render_target

    guard_output_root(config.output_root)  # render writes derived artifacts too
    failures = 0
    for target in targets:
        try:
            stats = render_target(config, target)
        except FileNotFoundError as exc:
            failures += 1
            print(f"WARN: render failed for {target.name}: {exc}", file=sys.stderr)
            continue
        missing = f", {len(stats.files_missing_summary)} without summaries" \
            if stats.files_missing_summary else ""
        print(
            f"rendered {target.name}: {stats.modules_written} module docs, architecture="
            f"{'yes' if stats.architecture_written else 'no'}{missing} → {config.out_dir(target)}"
        )
    return 1 if targets and failures == len(targets) else 0


def _cmd_run(config: MapConfig, targets: list[TargetConfig]) -> int:
    from forge.cartographer.config import settings
    from forge.cartographer.guards import run_guards
    from forge.cartographer.render import run_all

    run_guards(settings.openai_base_url, config.output_root)  # before any inference; no bypass
    exit_code, reports = run_all(config, settings, targets)
    for r in reports:
        line = "OK" if r.ok else f"FAILED — {r.error}"
        print(f"{r.name}: {line} ({r.wall_seconds:.0f}s)")
    print(f"report → {config.output_root / 'report.md'}")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = _load(args)
    if args.command == "targets":
        return _cmd_targets(config)
    targets = _selected(config, args)
    if args.command == "sync":
        return _cmd_sync(config, targets)
    if args.command == "structure":
        return _cmd_structure(config, targets)
    if args.command == "summarize":
        return _cmd_summarize(config, targets)
    if args.command == "render":
        return _cmd_render(config, targets)
    return _cmd_run(config, targets)


if __name__ == "__main__":
    raise SystemExit(main())
