"""The collision seat's charter and evidence rendering.

The seat answers ONE question: does anything upstream changed collide with the fork's
additive layer? Textual conflicts are already handled deterministically (git merge); the
seat exists for the semantic case — upstream renamed/refactored/behavior-changed something
the layer imports, wraps, or hooks, so the merge is clean but the fork is broken.

Diff-literacy rules mirror the epic gate's: the --stat manifest is the COMPLETE upstream
change set; full hunks are shown only where the risk concentrates (overlap files). A
finding must cite a specific file from the evidence; concerns that cannot be tied to a
file are notes, not findings — collision=true with zero cited findings is invalid.
"""

from __future__ import annotations

from forge.upstream_sync.models import LayerManifest

COLLISION_SEAT = """\
You review upstream merges into an ADDITIVE FORK: the fork adds files and lightly edits a
few upstream files; upstream evolves independently. The textual merge is already clean —
your job is the semantic risk a clean merge hides.

Decide: does any upstream change collide with the fork's additive layer?
Collisions look like:
- upstream renamed, moved, or deleted a function/type/file the layer's files import or wrap
- upstream changed the behavior or signature of something the layer hooks or extends
- upstream added a file at a path the layer also added (divergent implementations)
- upstream restructured wiring (routes, registries, config keys) the layer plugs into

Evidence rules (BINDING):
- The "Upstream change manifest" (--stat) is the COMPLETE upstream change set. A file not
  listed there is unchanged — do not speculate about unlisted files.
- Full diff hunks are provided only for the overlap files; judge the rest from the
  manifest, the commit log, and the layer file list.
- Every finding MUST cite one specific file that appears in the evidence. A concern you
  cannot tie to a file goes in "notes", never in "findings".
- collision=true requires at least one cited finding. When the evidence shows no
  collision, say collision=false — an empty findings list with vague worry is false.

Respond with ONLY a JSON object:
{"collision": true/false, "findings": [{"file": "path", "reason": "one sentence"}],
 "notes": "anything worth a human's eyes that is not a citable finding"}
"""


def _capped_list(items: list[str], cap: int = 200) -> str:
    shown = items[:cap]
    suffix = f"\n  ... and {len(items) - cap} more" if len(items) > cap else ""
    return "\n".join(f"  {item}" for item in shown) + suffix


def render_seat_evidence(
    *,
    layer: LayerManifest,
    upstream_log: str,
    upstream_stat: str,
    overlap: list[str],
    overlap_diff: str,
) -> str:
    """The seat's user message: layer ground truth, complete manifest, targeted hunks."""
    sections = [
        "## The fork's additive layer (computed from the merge-base — ground truth)",
        "Files the fork ADDED (upstream has no copy):",
        _capped_list(layer.added) or "  (none)",
        "Upstream files the fork MODIFIED:",
        _capped_list(layer.modified) or "  (none)",
        "",
        "## Upstream commits since the merge-base",
        upstream_log or "(none)",
        "",
        "## Upstream change manifest — the COMPLETE change set (--stat)",
        upstream_stat or "(none)",
    ]
    if overlap:
        sections += [
            "",
            "## Overlap files (upstream AND the fork both touch these) — full hunks",
            _capped_list(overlap),
            "",
            overlap_diff or "(diff unavailable)",
        ]
    else:
        sections += ["", "## Overlap files", "(none — upstream touched no fork-edited file)"]
    return "\n".join(sections)
