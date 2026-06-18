"""Large-PR digest pass: a size-guarded hybrid navigational digest of a big feature PR.

Unlike the review pass (fan out N reviewers, then synthesize), the digest wants a single coherent
reading guide — so it runs through a failover ``Pool`` (rotation: preferred → anthropic →
opencode_zen → local break-glass) and takes whichever model answers.

Size guard:
- diff <= digest_max_diff_chars → single pass (read the whole diff).
- larger → map-reduce: split into per-file chunks, summarize each (bounded concurrency, per-chunk
  failover), then synthesize the digest from the summaries (concat fallback if the reduce is down).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from agents.pr_review_ensemble.config import settings
from agents.pr_review_ensemble.diffsplit import DiffChunk, split_diff
from agents.pr_review_ensemble.models import DigestResult
from agents.pr_review_ensemble.prompts import (
    DIGEST_MAP_SYSTEM_PROMPT,
    DIGEST_REDUCE_SYSTEM_PROMPT,
    DIGEST_SYSTEM_PROMPT,
)
from agents.pr_review_ensemble.providers import ReviewerSlot, build_reviewer_slots, rotation_pool
from agents.shared.ensemble import Pool, Prompt


def build_digest_pool(slots: list[ReviewerSlot]) -> Pool:
    """A failover pool over active providers, strongest-first (same rotation as the aggregator)."""
    return rotation_pool(slots, role="digest", preferred=settings.aggregator_provider)


def _chunk_label(chunk: DiffChunk) -> str:
    return ", ".join(chunk.files) if chunk.files else "chunk"


async def _single_pass(diff_text: str, pr_ref: str, pool: Pool, base: DigestResult) -> DigestResult:
    user = f"Pull request: {pr_ref}\nDiff size: {base.diff_lines} lines\n\nDiff:\n{diff_text}"
    prompt = Prompt(system=DIGEST_SYSTEM_PROMPT, user=user, max_tokens=settings.digest_max_tokens)
    result = await pool.run(prompt, timeout=settings.per_provider_timeout_seconds)
    if not result.ok:
        return base.model_copy(update={"model": result.executor, "error": result.error})
    return base.model_copy(update={"digest": result.output, "model": result.executor})


async def _summarize_chunk(
    chunk: DiffChunk, pool: Pool, sem: asyncio.Semaphore
) -> tuple[str, bool]:
    """Map one chunk to a per-file summary. Returns (markdown_section, ok)."""
    async with sem:
        user = f"Files in this slice: {_chunk_label(chunk)}\n\n{chunk.text}"
        prompt = Prompt(
            system=DIGEST_MAP_SYSTEM_PROMPT, user=user, max_tokens=settings.digest_map_max_tokens
        )
        result = await pool.run(prompt, timeout=settings.per_provider_timeout_seconds)
    head = f"### {_chunk_label(chunk)}"
    if result.ok:
        return f"{head}\n\n{result.output}", True
    return f"{head}\n\n_[summary unavailable: {result.error}]_", False


async def _map_reduce(diff_text: str, pr_ref: str, pool: Pool, base: DigestResult) -> DigestResult:
    chunks = split_diff(diff_text, chunk_chars=settings.digest_chunk_chars)
    dropped = max(0, len(chunks) - settings.digest_max_chunks)
    chunks = chunks[: settings.digest_max_chunks]
    fields = {"strategy": "map_reduce", "chunks": len(chunks), "chunks_dropped": dropped}

    sem = asyncio.Semaphore(settings.digest_map_concurrency)
    mapped = await asyncio.gather(*(_summarize_chunk(c, pool, sem) for c in chunks))
    summaries = [section for section, _ in mapped]
    if not any(ok for _, ok in mapped):
        return base.model_copy(
            update={**fields, "error": f"all {len(chunks)} chunk summaries failed"}
        )

    body = "\n\n".join(summaries)
    note = (
        f"The diff ({base.diff_chars} chars) was too large to read whole; it was summarized per "
        f"file group across {len(chunks)} chunk(s) below."
    )
    if dropped:
        note += f" NOTE: {dropped} further chunk(s) were dropped to stay within limits."
    user = f"Pull request: {pr_ref}\n{note}\n\nPer-file summaries:\n\n{body}"
    prompt = Prompt(
        system=DIGEST_REDUCE_SYSTEM_PROMPT, user=user, max_tokens=settings.digest_max_tokens
    )
    result = await pool.run(prompt, timeout=settings.per_provider_timeout_seconds)
    if result.ok:
        return base.model_copy(update={**fields, "digest": result.output, "model": result.executor})

    # Reduce is down — still useful: hand back the per-file summaries with a clear note.
    fallback = (
        f"_(digest synthesis unavailable: {result.error}; showing per-file summaries)_\n\n{body}"
    )
    return base.model_copy(
        update={**fields, "digest": fallback, "model": "fallback:concat", "error": result.error}
    )


async def run_digest(
    *,
    diff_text: str,
    pr_ref: str,
    slots: list[ReviewerSlot] | None = None,
    pool: Pool | None = None,
) -> DigestResult:
    """Produce a navigational digest of a PR diff (single-pass or map-reduce by size)."""
    base = DigestResult(
        pr_ref=pr_ref,
        timestamp=datetime.now(UTC),
        diff_lines=diff_text.count("\n") + 1,
        diff_chars=len(diff_text),
    )
    if pool is None:
        pool = build_digest_pool(slots if slots is not None else build_reviewer_slots())
    if not pool.executors:
        return base.model_copy(update={"error": "no active providers for the digest pool"})

    if base.diff_chars <= settings.digest_max_diff_chars:
        return await _single_pass(diff_text, pr_ref, pool, base)
    return await _map_reduce(diff_text, pr_ref, pool, base)
