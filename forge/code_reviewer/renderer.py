from __future__ import annotations

from agents.code_reviewer.config import settings
from agents.code_reviewer.models import NightlyReport, RepoScores
from agents.shared.models.nous import EditorJsBlock
from agents.shared.renderer.blocks import checklist, header, paragraph


def _score_marker(score: int) -> str:
    """Return a visual indicator for extreme scores."""
    if score <= settings.score_alert_threshold:
        return " \u26a0\ufe0f"
    if score >= 9:
        return " \u2728"
    return ""


def _scores_markdown_table(
    rows: list[tuple[str, RepoScores]],
    *,
    show_repo_column: bool = True,
) -> list[str]:
    """Build a markdown table of scores."""
    lines: list[str] = []
    if show_repo_column:
        hdr = "| Repo | Security | Correctness | Error Handling | Performance | Overall |"
        sep = "|------|----------|-------------|----------------|-------------|---------|"
    else:
        hdr = "| Security | Correctness | Error Handling | Performance | Overall |"
        sep = "|----------|-------------|----------------|-------------|---------|"
    lines.append(hdr)
    lines.append(sep)
    for label, s in rows:
        cells = [
            f"{s.security}{_score_marker(s.security)}",
            f"{s.correctness}{_score_marker(s.correctness)}",
            f"{s.error_handling}{_score_marker(s.error_handling)}",
            f"{s.performance}{_score_marker(s.performance)}",
            f"{s.overall}{_score_marker(s.overall)}",
        ]
        if show_repo_column:
            lines.append(f"| {label} | {' | '.join(cells)} |")
        else:
            lines.append(f"| {' | '.join(cells)} |")
    return lines


def _aggregate_scores(report: NightlyReport) -> RepoScores | None:
    """Compute average scores across all scored repos."""
    scored = [r.scores for r in report.reviews if r.scores is not None]
    if not scored:
        return None
    n = len(scored)
    return RepoScores(
        security=round(sum(s.security for s in scored) / n),
        correctness=round(sum(s.correctness for s in scored) / n),
        error_handling=round(sum(s.error_handling for s in scored) / n),
        performance=round(sum(s.performance for s in scored) / n),
        overall=round(sum(s.overall for s in scored) / n),
    )


def render_markdown(report: NightlyReport) -> str:
    """Render the report as markdown for logging/debugging."""
    lines: list[str] = []
    lines.append(f"# Nightly Code Review — {report.date}")
    lines.append("")
    lines.append(
        f"**{report.repos_reviewed}** repos scanned, **{report.repos_with_changes}** with changes"
    )
    lines.append("")

    # Aggregate scores table
    agg = _aggregate_scores(report)
    if agg:
        lines.append("### Aggregate Scores")
        lines.append("")
        lines.extend(_scores_markdown_table([("Average", agg)], show_repo_column=False))
        lines.append("")

    lines.append(report.overall_summary)
    lines.append("")

    for review in report.reviews:
        lines.append(f"## {review.repo_name}")
        lines.append("")

        # Per-repo scores table
        if review.scores:
            lines.extend(
                _scores_markdown_table([(review.repo_name, review.scores)], show_repo_column=False)
            )
            lines.append("")

        lines.append(review.summary)
        lines.append("")

        if review.findings:
            for finding in review.findings:
                prefix = f"[{finding.severity.upper()}]"
                lines.append(f"- {prefix} `{finding.file_path}`: {finding.description}")
            lines.append("")
        else:
            lines.append("No issues found.")
            lines.append("")

    return "\n".join(lines)


def _scores_html_table(scores: RepoScores) -> str:
    """Build an inline HTML table for scores (used in Editor.js paragraph blocks)."""

    def _cell(val: int) -> str:
        marker = _score_marker(val)
        return f"<td>{val}{marker}</td>"

    return (
        "<table>"
        "<tr><th>Security</th><th>Correctness</th><th>Error Handling</th>"
        "<th>Performance</th><th>Overall</th></tr>"
        f"<tr>{_cell(scores.security)}{_cell(scores.correctness)}"
        f"{_cell(scores.error_handling)}{_cell(scores.performance)}"
        f"{_cell(scores.overall)}</tr>"
        "</table>"
    )


def render_blocks(report: NightlyReport) -> list[EditorJsBlock]:
    """Render the report as Editor.js blocks for the Nous daily note."""
    blocks: list[EditorJsBlock] = []

    blocks.append(paragraph(settings.review_marker))

    # Main header
    blocks.append(header("Nightly Code Review", level=2))

    # Stats line
    blocks.append(
        paragraph(
            f"<b>{report.repos_reviewed}</b> repos scanned, "
            f"<b>{report.repos_with_changes}</b> with changes"
        )
    )

    # Aggregate scores table
    agg = _aggregate_scores(report)
    if agg:
        blocks.append(header("Aggregate Scores", level=3))
        blocks.append(paragraph(_scores_html_table(agg)))

    # Overall summary
    blocks.append(paragraph(report.overall_summary))

    # Per-repo reviews
    for review in report.reviews:
        blocks.append(header(review.repo_name, level=3))

        # Per-repo scores
        if review.scores:
            blocks.append(paragraph(_scores_html_table(review.scores)))

        blocks.append(paragraph(review.summary))

        if review.findings:
            # Split into issues (unchecked) and positives (checked)
            issue_items: list[str] = []
            positive_items: list[str] = []

            for finding in review.findings:
                prefix = f"[{finding.severity.upper()}]"
                text = f"{prefix} <code>{finding.file_path}</code>: {finding.description}"

                if finding.severity == "positive":
                    positive_items.append(text)
                else:
                    issue_items.append(text)

            if issue_items:
                blocks.append(
                    EditorJsBlock(
                        type="checklist",
                        data={"items": [{"text": item, "checked": False} for item in issue_items]},
                    )
                )

            if positive_items:
                blocks.append(
                    EditorJsBlock(
                        type="checklist",
                        data={
                            "items": [{"text": item, "checked": True} for item in positive_items]
                        },
                    )
                )

    return blocks
