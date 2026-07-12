"""Compute the additive-layer manifest: what the fork owns relative to the merge-base.

``git diff --name-status -M <merge_base>..<local_tip>`` classifies every fork-side change:
A → the fork created the file (upstream has no copy); M/D → the fork edited or removed an
upstream file; R → a rename, whose OLD path is treated as fork-modified (upstream changes
to it will collide) and whose NEW path is fork-owned. The manifest is the collision seat's
ground truth — computed, never guessed.
"""

from __future__ import annotations

from pathlib import Path

from forge.upstream_sync.gitops import git
from forge.upstream_sync.models import LayerManifest


def compute_layer(repo: Path, merge_base: str, local_tip: str) -> LayerManifest:
    out = git(repo, "diff", "--name-status", "-M", f"{merge_base}..{local_tip}")
    added: set[str] = set()
    modified: set[str] = set()
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0]
        if status.startswith(("R", "C")) and len(parts) >= 3:
            # Rename/copy: new path is fork-owned; a rename's old path stays collision-prone.
            added.add(parts[2])
            if status.startswith("R"):
                modified.add(parts[1])
        elif status.startswith("A"):
            added.add(parts[1])
        elif status.startswith(("M", "D", "T")):
            modified.add(parts[1])
    return LayerManifest(added=sorted(added), modified=sorted(modified))
