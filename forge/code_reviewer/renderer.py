from __future__ import annotations

from agents.code_reviewer.models import NightlyReport
from agents.shared.models.nous import EditorJsBlock
from agents.shared.renderer.blocks import checklist, header, paragraph


def render_markdown(report: NightlyReport) -> str:
    """Render the report as markdown for logging/debugging."""
    lines: list[str] = []
    lines.append(f"# Nightly Code Review — {report.date}")
    lines.append("")
    lines.append(
        f"**{report.repos_reviewed}** repos scanned, "
        f"**{report.repos_with_changes}** with changes"
    )
    lines.append("")
    lines.append(report.overall_summary)
    lines.append("")

    for review in report.reviews:
        lines.append(f"## {review.repo_name}")
        lines.append("")
        lines.append(review.summary)
        lines.append("")

        if review.findings:
            for finding in review.findings:
                prefix = f"[{finding.severity.upper()}]"
                lines.append(
                    f"- {prefix} `{finding.file_path}`: {finding.description}"
                )
            lines.append("")
        else:
            lines.append("No issues found.")
            lines.append("")

    return "\n".join(lines)


def render_blocks(report: NightlyReport) -> list[EditorJsBlock]:
    """Render the report as Editor.js blocks for the Nous daily note."""
    blocks: list[EditorJsBlock] = []

    # Hidden marker for idempotency
    from agents.code_reviewer.config import settings

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

    # Overall summary
    blocks.append(paragraph(report.overall_summary))

    # Per-repo reviews
    for review in report.reviews:
        blocks.append(header(review.repo_name, level=3))
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
                        data={
                            "items": [
                                {"text": item, "checked": False}
                                for item in issue_items
                            ]
                        },
                    )
                )

            if positive_items:
                blocks.append(
                    EditorJsBlock(
                        type="checklist",
                        data={
                            "items": [
                                {"text": item, "checked": True}
                                for item in positive_items
                            ]
                        },
                    )
                )

    return blocks
