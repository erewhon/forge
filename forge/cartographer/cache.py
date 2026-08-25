"""Content-addressed summary cache under ``<output_root>/cache``.

File summaries are keyed by the file's blob hash, so the cache is shared across every target:
two checkouts of the same repo on different VMs reuse each other's entries for free, and an
unchanged file is never re-summarized. Rollups and architecture docs are keyed by digests over
their inputs' keys, giving the same property one level up — touch one file and exactly that
file, its module rollup, and the architecture doc recompute.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path


def rollup_digest(module_name: str, member_blob_shas: Iterable[str]) -> str:
    payload = module_name + "\n" + "\n".join(sorted(member_blob_shas))
    return hashlib.sha256(payload.encode()).hexdigest()


def architecture_digest(rollup_digests: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(rollup_digests)).encode()).hexdigest()


class SummaryCache:
    def __init__(self, output_root: Path) -> None:
        self.root = output_root / "cache"

    def file_summary(self, blob_sha: str) -> Path:
        return self.root / "summaries" / f"{blob_sha}.md"

    def rollup(self, digest: str) -> Path:
        return self.root / "rollups" / f"{digest}.md"

    def architecture(self, digest: str) -> Path:
        return self.root / "arch" / f"{digest}.md"

    @staticmethod
    def get(path: Path) -> str | None:
        try:
            text = path.read_text()
        except OSError:
            return None
        return text or None

    @staticmethod
    def put(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
