from __future__ import annotations

import json
from pathlib import Path

from agents.pr_review_ensemble.config import settings
from agents.pr_review_ensemble.models import DigestResult, EnsembleResult


def log_run(result: EnsembleResult, *, log_path: Path | None = None) -> Path:
    """Append one JSONL record for the run. Returns the log path."""
    target = log_path or settings.log_path
    target.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "pass": "review",
        "timestamp": result.timestamp.isoformat(),
        "pr_ref": result.pr_ref,
        "diff_lines": result.diff_lines,
        "quorum_state": result.quorum_state,
        "quorum_floor": result.quorum_floor,
        "providers_attempted": result.providers_attempted,
        "providers_succeeded": result.providers_succeeded,
        "aggregator_provider": result.aggregator_provider,
        "aggregator_used_fallback": result.aggregator_used_fallback,
        "per_provider": [
            {
                "provider": r.provider,
                "model": r.model,
                "status": r.status,
                "latency_ms": r.latency_ms,
                "error_message": r.error_message,
                "response_chars": len(r.response_text) if r.response_text else 0,
            }
            for r in result.reviews
        ],
    }

    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    return target


def log_digest(result: DigestResult, *, log_path: Path | None = None) -> Path:
    """Append one JSONL record for a digest run (shares the log with review runs via `pass`)."""
    target = log_path or settings.log_path
    target.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "pass": "digest",
        "timestamp": result.timestamp.isoformat(),
        "pr_ref": result.pr_ref,
        "diff_lines": result.diff_lines,
        "diff_chars": result.diff_chars,
        "model": result.model,
        "strategy": result.strategy,
        "chunks": result.chunks,
        "chunks_dropped": result.chunks_dropped,
        "ok": result.digest is not None,
        "error": result.error,
        "digest_chars": len(result.digest) if result.digest else 0,
    }

    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    return target
