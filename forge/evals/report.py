"""Scorecard report rendering and persistence.

Provides ``render_scorecard`` (markdown table output) and
``write_scorecard`` (persists ``scorecard.json`` + ``scorecard.md``
under ``runs_dir/<model>/``).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from forge.evals.models import Scorecard


def render_scorecard(sc: Scorecard) -> str:
    """Render a *scorecard* as markdown.

    Per-step table: cases, pass-rate, holdout pass-rate, mean score, error
    repeats.  Overall line at the bottom.
    """
    lines: list[str] = []
    lines.append(f"# Scorecard: {sc.model}")
    lines.append("")
    lines.append(f"Timestamp: {sc.timestamp}")
    lines.append("")

    for step_score in sc.steps:
        lines.append(f"## Step: `{step_score.step}`")
        lines.append("")

        header = "| Case | Pass Rate | Holdout Pass Rate | Mean Score | Error Repeats |"
        sep = "|-------|-----------|-------------------|------------|---------------|"
        lines.append(header)
        lines.append(sep)

        for case_score in step_score.cases:
            case_id = case_score.case_id
            pass_rate = f"{case_score.passed_majority:.0%}"
            mean_score = f"{case_score.mean_score:.4f}"

            # Holdout pass rate
            if case_score.holdout:
                holdout_str = "N/A (this IS holdout)"
            else:
                holdout_str = "N/A"

            error_repeats = sum(1 for r in case_score.repeats if r.error is not None)
            lines.append(
                f"| {case_id} | {pass_rate} | {holdout_str} | {mean_score} | {error_repeats} |"
            )

        lines.append("")
        lines.append(
            f"**Step pass-rate:** {step_score.pass_rate:.0%}  "
            f"**Holdout pass-rate:** "
            f"{step_score.holdout_pass_rate:.0%}"
            if step_score.holdout_pass_rate is not None
            else f"N/A  **Error repeats:** {step_score.error_repeats}"
        )
        lines.append("")

    lines.append(f"**Overall pass-rate:** {sc.overall_pass_rate:.0%}")
    lines.append("")

    if sc.notes:
        lines.append(sc.notes)
        lines.append("")

    return "\n".join(lines)


def write_scorecard(sc: Scorecard, runs_dir: Path) -> Path:
    """Persist *scorecard* as ``scorecard.json`` + ``scorecard.md``.

    Files are written under ``runs_dir/<UTC-stamp>-<model>/`` so successive runs
    never overwrite each other — the run history is part of the record.

    Returns
    -------
    Path
        The path to the ``scorecard.json`` file.
    """
    stamp = datetime.fromisoformat(sc.timestamp).strftime("%Y%m%dT%H%M%SZ")
    output_dir = runs_dir / f"{stamp}-{sc.model.replace('/', '-')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write JSON
    json_path = output_dir / "scorecard.json"
    json_path.write_text(sc.model_dump_json(indent=2), encoding="utf-8")

    # Write markdown
    md_path = output_dir / "scorecard.md"
    md_path.write_text(render_scorecard(sc), encoding="utf-8")

    return json_path
