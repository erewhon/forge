"""Assemble the per-target deliverables and the run report; orchestrate the full pipeline.

``render_target`` turns the structural index + the summarizer's manifest into the documents an
agent (or a person) actually reads: ``modules/<name>.md``, ``architecture.md``, a refreshed
``map.md``. ``run_all`` chains sync → structure → summarize → render per target with hard
failure isolation — one dead VM or broken repo never sinks the others — and writes
``report.md`` naming everything that was skipped or failed. No silent caps anywhere.

Guards note: callers of ``run_all`` must run :func:`forge.cartographer.guards.run_guards`
first (the CLI does); this module adds no bypasses.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from forge.cartographer.cache import SummaryCache
from forge.cartographer.config import CartographerSettings
from forge.cartographer.models import MapConfig, TargetConfig, TargetInfo
from forge.cartographer.structural import StructuralResult, build_index, render_map, write_outputs
from forge.cartographer.summarize import CompleteFn, SummarizeStats, load_index, summarize_target
from forge.cartographer.sync import (
    Runner,
    _default_runner,
    read_sync_detail,
    read_sync_state,
    sync_target,
)


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "module"


@dataclass
class RenderStats:
    modules_written: int = 0
    files_with_summary: int = 0
    files_missing_summary: list[str] = field(default_factory=list)
    architecture_written: bool = False


@dataclass
class TargetReport:
    name: str
    ok: bool = True
    error: str = ""
    sync_ok: bool = True
    stale: bool = False
    parsed: int = 0
    summarize: SummarizeStats | None = None
    render: RenderStats | None = None
    wall_seconds: float = 0.0
    consecutive_failures: int = 0
    returncode: int | None = None
    stderr_tail: str = ""
    synced_at: datetime | None = None


def _mirror_age_label(synced_at: datetime | None) -> str:
    """How old a stale mirror is, for the run report: the last successful sync."""
    if synced_at is None:
        return "never synced"
    age_days = (datetime.now(UTC) - synced_at).days
    if age_days == 0:
        return "stale mirror, synced today"
    return f"stale mirror, {age_days} days old"


def _load_manifest(config: MapConfig, target: TargetConfig) -> dict:
    path = config.out_dir(target) / "summaries.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no summaries for target '{target.name}' — run `forge map summarize` first"
        )
    return json.loads(path.read_text())


def _cache_text(config: MapConfig, rel_file: str | None) -> str | None:
    if not rel_file:
        return None
    return SummaryCache.get(config.output_root / rel_file)


def render_target(config: MapConfig, target: TargetConfig) -> RenderStats:
    index = load_index(config, target)
    manifest = _load_manifest(config, target)
    out = config.out_dir(target)
    stats = RenderStats()

    # Refresh map.md from the index so it reflects the current sync state line.
    synced_at, stale = read_sync_state(config, target)
    index.target.synced_at, index.target.stale = synced_at, stale
    parsed = [f for f in index.files if not f.skipped]
    structural = StructuralResult(
        index=index, parsed=len(parsed), skipped=[f.path for f in index.files if f.skipped]
    )
    (out / "map.md").write_text(render_map(structural))

    modules_dir = out / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)
    file_meta = manifest.get("files", {})
    for module in index.modules:
        entries = [f for f in index.files if f.module == module.name]
        if not entries:
            continue
        lines = [f"# {module.name} (`{module.path}`)", ""]
        rollup = _cache_text(
            config, manifest.get("modules", {}).get(module.name, {}).get("rollup_file")
        )
        lines += [rollup or "*(rollup not yet generated — run `forge map summarize`)*", ""]
        for entry in entries:
            lines.append(f"## `{entry.path}`" + (" *(entry point)*" if entry.entry_point else ""))
            summary = _cache_text(config, file_meta.get(entry.blob_sha, {}).get("summary_file"))
            if summary:
                stats.files_with_summary += 1
                lines.append(summary)
            else:
                stats.files_missing_summary.append(entry.path)
                lines.append("*(no summary yet)*")
            if entry.symbols:
                lines.append(f"\n- symbols: {'; '.join(entry.symbols)}")
            if entry.imports:
                lines.append(f"- imports: {'; '.join(entry.imports)}")
            lines.append("")
        (modules_dir / f"{_slug(module.name)}.md").write_text("\n".join(lines))
        stats.modules_written += 1

    arch_meta = manifest.get("architecture") or {}
    arch = _cache_text(config, arch_meta.get("file"))
    header = f"# Architecture — {target.name}\n\n"
    (out / "architecture.md").write_text(
        header + (arch or "*(architecture doc not yet generated — run `forge map summarize`)*\n")
    )
    stats.architecture_written = arch is not None
    return stats


# --- full pipeline -------------------------------------------------------------------------------


def run_all(
    config: MapConfig,
    settings: CartographerSettings,
    targets: list[TargetConfig],
    sweep: CompleteFn | None = None,
    synthesis: CompleteFn | None = None,
    runner: Runner = _default_runner,
) -> tuple[int, list[TargetReport]]:
    reports: list[TargetReport] = []
    for target in targets:
        report = TargetReport(name=target.name)
        start = time.monotonic()
        try:
            sync = sync_target(config, target, runner)
            report.sync_ok = sync.ok
            report.returncode = sync.returncode
            report.stderr_tail = sync.stderr_tail
            if not sync.ok:
                # The run report is read later; the journal is where a failing nightly is noticed.
                print(
                    f"WARN: sync failed for {target.name} (exit {sync.returncode}): {sync.message}",
                    file=sys.stderr,
                )
            detail = read_sync_detail(config, target)
            report.stale = detail.stale
            report.consecutive_failures = detail.consecutive_failures
            report.synced_at = detail.synced_at
            if not sync.ok and detail.synced_at is None:
                # Never synced successfully: the mirror dir exists but is empty — an "OK" empty
                # map here would silently read as coverage. Fail the target instead.
                raise RuntimeError(f"sync failed and no previous mirror exists ({sync.message})")
            info = TargetInfo(
                name=target.name,
                host=target.host,
                path=target.path,
                synced_at=detail.synced_at,
                stale=detail.stale,
            )
            structural = build_index(config, target, info)
            write_outputs(config, target, structural)
            report.parsed = structural.parsed
            report.summarize = summarize_target(config, settings, target, sweep, synthesis)
            report.render = render_target(config, target)
        except Exception as exc:  # noqa: BLE001 - per-target isolation boundary
            report.ok = False
            report.error = f"{type(exc).__name__}: {exc}"
        report.wall_seconds = time.monotonic() - start
        reports.append(report)
    _write_report(config, reports)
    exit_code = 1 if targets and not any(r.ok for r in reports) else 0
    return exit_code, reports


def _write_report(config: MapConfig, reports: list[TargetReport]) -> None:
    lines = [f"# forge map run — {datetime.now(UTC):%Y-%m-%d %H:%M UTC}", ""]

    # Check for attention: any target with >= 2 consecutive failures
    attention_targets = [r for r in reports if r.consecutive_failures >= 2]
    if attention_targets:
        for r in attention_targets:
            lines.append(
                f"ATTENTION: {r.name} has failed sync {r.consecutive_failures} runs in a row"
            )
        lines.append("")

    for r in reports:
        status = "OK" if r.ok else f"FAILED — {r.error}"
        lines.append(f"## {r.name}: {status}")
        if not r.ok:
            lines.append("")
            continue
        if not r.sync_ok:
            lines.append(
                f"- ⚠ sync failed — mapped from the previous mirror "
                f"({_mirror_age_label(r.synced_at)})"
            )
            if r.returncode is not None:
                lines.append(f"- rsync exit code: {r.returncode}")
            if r.stderr_tail:
                lines.append("- stderr tail:")
                lines.append("  ```")
                lines.extend(f"  {line}" for line in r.stderr_tail.splitlines())
                lines.append("  ```")
        s = r.summarize
        lines.append(f"- files parsed: {r.parsed}")
        if s:
            lines.append(
                f"- summaries: {s.summarized} new, {s.cache_hits} cached, "
                f"{s.rollups_built} rollups, {s.llm_calls} LLM calls"
            )
            for entry in s.skipped:
                lines.append(f"- skipped: {entry}")
            for entry in s.failed:
                lines.append(f"- FAILED (will retry next run): {entry}")
            for entry in dict.fromkeys(s.errors):
                lines.append(f"- error: {entry}")
        if r.render:
            lines.append(
                f"- rendered: {r.render.modules_written} module docs, architecture="
                f"{'yes' if r.render.architecture_written else 'NO'}"
            )
            for entry in r.render.files_missing_summary:
                lines.append(f"- no summary yet: {entry}")
        lines.append(f"- wall: {r.wall_seconds:.0f}s")
        lines.append("")
    config.output_root.mkdir(parents=True, exist_ok=True)
    (config.output_root / "report.md").write_text("\n".join(lines))
