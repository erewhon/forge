"""Render a ParallelEditResult to markdown."""

from __future__ import annotations

from forge.parallel_edit.models import (
    DimensionScores,
    EditRun,
    JudgeVerdict,
    ParallelEditResult,
)

_DIMENSIONS = (
    "prompt_fidelity",
    "correctness",
    "scope_discipline",
    "code_quality",
    "completeness",
)


def _format_run_header(run: EditRun) -> str:
    latency = f" — {run.latency_ms} ms" if run.latency_ms is not None else ""
    stat = run.diff_stat
    stat_text = f"{stat.files_changed} files / +{stat.insertions} / -{stat.deletions}"
    return (
        f"### Candidate {run.label} ({run.model}) — _{run.status}_{latency}\n\n"
        f"Workspace: `{run.workspace_path}`  \n"
        f"Diff stat: {stat_text}"
    )


def _format_run_failure_detail(run: EditRun) -> str:
    lines: list[str] = []
    if run.error_message:
        lines.append(f"**Error:** {run.error_message}")
    if run.returncode is not None and run.returncode != 0:
        lines.append(f"**Exit code:** {run.returncode}")
    if run.stderr_tail.strip():
        lines.append("**stderr (tail):**")
        lines.append("```")
        lines.append(run.stderr_tail.rstrip())
        lines.append("```")
    if run.stdout_tail.strip():
        lines.append("**stdout (tail):**")
        lines.append("```")
        lines.append(run.stdout_tail.rstrip())
        lines.append("```")
    return "\n".join(lines)


def _scores_table(scores: dict[str, DimensionScores]) -> str:
    labels = sorted(scores.keys())
    header = "| Dimension | " + " | ".join(labels) + " |"
    sep = "|---|" + "|".join("---" for _ in labels) + "|"
    rows = [header, sep]
    for dim in _DIMENSIONS:
        cells = [getattr(scores[label], dim) for label in labels]
        rows.append(f"| {dim} | " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(rows)


def _format_verdict(verdict: JudgeVerdict, judge_model: str) -> str:
    lines: list[str] = []
    lines.append(f"## Verdict — winner: **{verdict.winner}** _(judge: {judge_model})_")
    lines.append("")
    if verdict.summary:
        lines.append(verdict.summary)
        lines.append("")
    lines.append("### Scores")
    lines.append("")
    lines.append(_scores_table(verdict.scores))
    lines.append("")
    if verdict.per_file_notes:
        lines.append("### Per-file comparison")
        lines.append("")
        lines.append("| File | Best | Note |")
        lines.append("|---|---|---|")
        for note in verdict.per_file_notes:
            cell = note.note.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{note.file}` | {note.best} | {cell} |")
        lines.append("")
    if verdict.recommendation:
        lines.append("### Recommendation")
        lines.append("")
        lines.append(verdict.recommendation)
    return "\n".join(lines)


def render_markdown(result: ParallelEditResult) -> str:
    lines: list[str] = []
    lines.append("# Parallel Edit Comparison")
    lines.append("")
    lines.append(f"**Repo:** `{result.repo_path}`  ")
    lines.append(f"**Base revision:** `{result.base_rev}`  ")
    lines.append(f"**Generated:** {result.timestamp.isoformat()}  ")
    lines.append(f"**Candidates:** {len(result.runs)}")
    lines.append("")
    lines.append("## Prompt")
    lines.append("")
    lines.append("```")
    lines.append(result.prompt.rstrip())
    lines.append("```")
    lines.append("")

    if result.verdict is not None:
        lines.append(_format_verdict(result.verdict, result.judge_model or "unknown"))
        lines.append("")
    else:
        lines.append("## Verdict")
        lines.append("")
        lines.append(f"_No verdict produced — {result.judge_error or 'unknown reason'}._")
        lines.append("")

    lines.append("## Candidate runs")
    lines.append("")
    for run in result.runs:
        lines.append(_format_run_header(run))
        lines.append("")
        if run.status in ("timeout", "error"):
            detail = _format_run_failure_detail(run)
            if detail:
                lines.append(detail)
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"
