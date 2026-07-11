"""Redundancy report mode: ask a model which dependencies overlap in purpose.

Read-only sub-mode — no bump loop, no writes, no task emission. Prints a markdown
report to stdout and exits 0 regardless of cluster count.

Designed as the candidate-list generator for a future AI-vendoring epic; the deps-v2
brief deliberately drops AI vendoring from scope.
"""

from __future__ import annotations

from pathlib import Path

from forge.dependabot.models import BumpCandidate, RedundancyCluster, RedundancyReport
from forge.dependabot.scan import scan_outdated
from forge.shared.llm import LLMConfig, complete, extract_json


def build_redundancy_prompt(deps: list[BumpCandidate]) -> tuple[str, str]:
    """Return (system_prompt, user_message) asking the model to identify overlapping-purpose
    clusters among the scanned direct dependencies.

    The prompt lists every dependency name + version so the model can judge semantic overlap.
    """
    dep_lines = "\n".join(f"- {d.name} {d.current}" for d in deps)
    system = (
        "You are a dependency-audit assistant. Your job is to identify libraries that serve the "
        "same or overlapping purpose in a project and recommend consolidation."
    )
    user = (
        f"Given these direct dependencies, identify any clusters of overlapping-purpose packages. "
        f"For each cluster state the shared purpose, list all overlapping packages, pick the one "
        f"to keep (best maintained, most widely used, or best fit), and write a short migration "
        f"note describing how to consolidate to the chosen package.\n\n"
        f"Direct dependencies:\n{dep_lines}\n\n"
        "Respond with ONLY a JSON object: "
        '{"clusters": [{"purpose": "...", "packages": ["..."], "keep": "...", '
        '"migration_note": "..."}]}'
    )
    return system, user


def call_model(
    deps: list[BumpCandidate],
    cfg: LLMConfig | None = None,
) -> RedundancyReport:
    """Ask the configured LLM for redundancy clusters and return a parsed RedundancyReport.

    When ``cfg`` is None the default ``LLMConfig(openai="openai")`` is used (points at the
    local LiteLLM router where the ``coder`` alias resolves).
    """
    cfg = cfg or LLMConfig(backend="openai")
    system, user = build_redundancy_prompt(deps)
    raw = complete(cfg, system=system, user_message=user, model="coder", max_tokens=4096)
    parsed = extract_json(raw)
    if not parsed or "clusters" not in parsed:
        return RedundancyReport(clusters=[])
    clusters = []
    for c in parsed["clusters"]:
        clusters.append(
            RedundancyCluster(
                purpose=c.get("purpose", ""),
                packages=c.get("packages", []),
                keep=c.get("keep", ""),
                migration_note=c.get("migration_note", ""),
            )
        )
    return RedundancyReport(clusters=clusters)


def render_report(report: RedundancyReport, deps: list[BumpCandidate]) -> str:
    """Render a *RedundancyReport* as a markdown string ready for stdout."""
    lines = ["# Dependency Redundancy Report"]
    lines.append("")
    lines.append(f"**Dependencies scanned:** {len(deps)}")
    lines.append("")

    if not report.clusters:
        lines.append(
            "The model found no overlapping-purpose clusters among the scanned dependencies. "
            "Each library appears to serve a distinct role."
        )
        lines.append("")
        return "\n".join(lines)

    lines.append(f"**Clusters found:** {len(report.clusters)}")
    lines.append("")

    for i, cluster in enumerate(report.clusters, 1):
        lines.append(f"## Cluster {i}: {cluster.purpose}")
        lines.append("")
        lines.append(f"**Packages:** {', '.join(cluster.packages)}")
        lines.append("")
        lines.append(f"**Keep:** `{cluster.keep}`")
        lines.append("")
        lines.append(f"**Migration:** {cluster.migration_note}")
        lines.append("")

    return "\n".join(lines)


def redundancy_report(
    repo_path: Path,
    *,
    cfg: LLMConfig | None = None,
) -> tuple[RedundancyReport, list[BumpCandidate]]:
    """Run the full read-only redundancy report: scan → ask model → render (returned, not printed).

    Returns ``(report, deps)`` so the caller (main.py) controls stdout output.
    """
    deps = scan_outdated(repo_path)
    report = call_model(deps, cfg=cfg)
    return report, deps
