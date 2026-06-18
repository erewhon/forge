"""Split a unified diff into reviewer-sized chunks for the digest's map step.

Pure and deterministic (no LLM): segment a git unified diff by file, pack small files together up
to a char budget, and split a single oversized file by hunk (truncating a single giant hunk as a
last resort). Each chunk stays self-describing (its file header is preserved) so a summarizer can
read it in isolation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_FILE_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)
_TRUNCATED = "\n[... hunk truncated to fit the digest chunk budget ...]\n"


@dataclass
class DiffChunk:
    files: list[str] = field(default_factory=list)
    text: str = ""


def file_segments(diff_text: str) -> list[DiffChunk]:
    """Split the diff at each `diff --git` header. Text before the first header (rare) is kept."""
    starts = [m.start() for m in _FILE_RE.finditer(diff_text)]
    if not starts:
        return [DiffChunk(files=[], text=diff_text)] if diff_text.strip() else []

    segments: list[DiffChunk] = []
    if starts[0] > 0:
        preamble = diff_text[: starts[0]]
        if preamble.strip():
            segments.append(DiffChunk(files=[], text=preamble))

    bounds = starts + [len(diff_text)]
    for i, start in enumerate(starts):
        text = diff_text[start : bounds[i + 1]]
        header = _FILE_RE.match(text)
        path = header.group(2) if header else "(unknown)"
        segments.append(DiffChunk(files=[path], text=text))
    return segments


def _split_oversized_file(seg: DiffChunk, chunk_chars: int) -> list[DiffChunk]:
    """Split one file segment that exceeds the budget into per-hunk pieces, repeating its header."""
    hunk_starts = [m.start() for m in re.finditer(r"^@@ ", seg.text, re.MULTILINE)]
    if not hunk_starts:
        return [DiffChunk(files=seg.files, text=seg.text[:chunk_chars] + _TRUNCATED)]

    header = seg.text[: hunk_starts[0]]
    bounds = hunk_starts + [len(seg.text)]
    pieces: list[DiffChunk] = []
    cur = header
    for i, hs in enumerate(hunk_starts):
        hunk = seg.text[hs : bounds[i + 1]]
        if len(header) + len(hunk) > chunk_chars:  # a single hunk too big on its own
            if cur != header:
                pieces.append(DiffChunk(files=seg.files, text=cur))
            budget = max(0, chunk_chars - len(header) - len(_TRUNCATED))
            pieces.append(DiffChunk(files=seg.files, text=header + hunk[:budget] + _TRUNCATED))
            cur = header
            continue
        if len(cur) + len(hunk) > chunk_chars and cur != header:
            pieces.append(DiffChunk(files=seg.files, text=cur))
            cur = header
        cur += hunk
    if cur != header:
        pieces.append(DiffChunk(files=seg.files, text=cur))
    return pieces


def _pack(units: list[DiffChunk], chunk_chars: int) -> list[DiffChunk]:
    """Greedily merge adjacent units (each already <= budget) up to the budget."""
    chunks: list[DiffChunk] = []
    for unit in units:
        if chunks and len(chunks[-1].text) + len(unit.text) <= chunk_chars:
            chunks[-1].text += unit.text
            chunks[-1].files += unit.files
        else:
            chunks.append(DiffChunk(files=list(unit.files), text=unit.text))
    return chunks


def split_diff(diff_text: str, *, chunk_chars: int) -> list[DiffChunk]:
    """Split a unified diff into chunks no larger than ~chunk_chars, preserving file structure."""
    units: list[DiffChunk] = []
    for seg in file_segments(diff_text):
        if len(seg.text) <= chunk_chars:
            units.append(seg)
        else:
            units.extend(_split_oversized_file(seg, chunk_chars))
    return _pack(units, chunk_chars)
