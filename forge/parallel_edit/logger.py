"""JSONL log for parallel_edit runs."""

from __future__ import annotations

import json
from pathlib import Path

from agents.parallel_edit.config import settings
from agents.parallel_edit.models import ParallelEditResult


def log_run(result: ParallelEditResult, *, log_path: Path | None = None) -> Path:
    """Append a single JSONL record for this run. Returns the log path."""
    target = log_path or settings.log_path
    target.parent.mkdir(parents=True, exist_ok=True)

    record: dict = {
        "timestamp": result.timestamp.isoformat(),
        "repo_path": str(result.repo_path),
        "base_rev": result.base_rev,
        "prompt_chars": len(result.prompt),
        "candidates": [
            {
                "label": r.label,
                "model": r.model,
                "status": r.status,
                "latency_ms": r.latency_ms,
                "returncode": r.returncode,
                "files_changed": r.diff_stat.files_changed,
                "insertions": r.diff_stat.insertions,
                "deletions": r.diff_stat.deletions,
                "diff_chars": len(r.diff_text),
                "error_message": r.error_message,
                "workspace_path": str(r.workspace_path),
            }
            for r in result.runs
        ],
        "judge_model": result.judge_model,
        "judge_error": result.judge_error,
        "verdict": (
            {
                "winner": result.verdict.winner,
                "scores": {
                    label: scores.model_dump() for label, scores in result.verdict.scores.items()
                },
                "per_file_count": len(result.verdict.per_file_notes),
                "summary_chars": len(result.verdict.summary),
            }
            if result.verdict is not None
            else None
        ),
    }

    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    return target
