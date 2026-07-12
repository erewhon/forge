"""The collision seat: one LLM call, validated hard, fail-closed for --auto-merge only.

Validation enforces the charter mechanically: findings citing files that do not appear in
the upstream change set or the layer are DEMOTED to notes (the diff-literacy lesson — an
uncited claim is an opinion), and collision=true with zero surviving findings downgrades
to false-with-notes. Any seat failure returns ``collision=None`` — unknown blocks
--auto-merge but never the default branch push.
"""

from __future__ import annotations

from forge.shared.llm import LLMConfig, complete, extract_json
from forge.upstream_sync.config import settings
from forge.upstream_sync.models import CollisionFinding, CollisionVerdict, LayerManifest
from forge.upstream_sync.prompts import COLLISION_SEAT, render_seat_evidence


def collision_verdict(
    *,
    layer: LayerManifest,
    upstream_files: list[str],
    upstream_log: str,
    upstream_stat: str,
    overlap: list[str],
    overlap_diff: str,
    cfg: LLMConfig | None = None,
) -> CollisionVerdict:
    """Ask the seat and validate its answer against the evidence universe."""
    if not settings.seat_enabled:
        return CollisionVerdict(collision=None, notes="collision seat disabled by settings")

    user = render_seat_evidence(
        layer=layer,
        upstream_log=upstream_log,
        upstream_stat=upstream_stat,
        overlap=overlap,
        overlap_diff=overlap_diff,
    )
    try:
        raw = complete(
            cfg
            or LLMConfig(
                backend="openai",
                openai_base_url=settings.openai_base_url,
                openai_api_key=settings.openai_api_key,
            ),
            system=COLLISION_SEAT,
            user_message=user,
            model=settings.seat_model,
            max_tokens=settings.seat_max_tokens,
        )
        parsed = extract_json(raw)
    except Exception as e:  # noqa: BLE001 — any seat failure is an unknown verdict, not a crash
        return CollisionVerdict(collision=None, notes=f"collision seat unavailable: {e}")

    if not parsed or "collision" not in parsed:
        return CollisionVerdict(collision=None, notes="collision seat returned unparseable output")

    citable = set(upstream_files) | set(layer.added) | set(layer.modified)
    findings: list[CollisionFinding] = []
    demoted: list[str] = []
    for f in parsed.get("findings") or []:
        file, reason = str(f.get("file", "")), str(f.get("reason", ""))
        if file in citable:
            findings.append(CollisionFinding(file=file, reason=reason))
        else:
            demoted.append(f"{file or '(no file)'}: {reason}")

    notes = str(parsed.get("notes", ""))
    if demoted:
        joined = "; ".join(demoted)
        notes = f"{notes} [uncited, demoted from findings: {joined}]".strip()

    collision = bool(parsed["collision"])
    if collision and not findings:
        # The charter says true requires a citation; an uncitable true is worry, not evidence.
        collision = False
        notes = f"{notes} [collision claimed without a citable finding — downgraded]".strip()

    return CollisionVerdict(collision=collision, findings=findings, notes=notes)
