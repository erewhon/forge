"""The LLM stage: file summaries → module rollups → architecture doc, cache-first.

Reads the structural ``index.json`` (the sync/structural stage's contract), fans file
summaries out to the sweep tier with bounded concurrency, then builds rollups and the
per-target architecture doc on the synthesis tier. Every artifact is cache-addressed
(:mod:`forge.cartographer.cache`), so steady-state runs cost only what changed.

Callers are responsible for running the disclosure guards first (``main`` does; so must any
later orchestration stage).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from forge.cartographer.cache import SummaryCache, architecture_digest, rollup_digest
from forge.cartographer.config import CartographerSettings
from forge.cartographer.models import FileEntry, MapConfig, RepoIndex, TargetConfig
from forge.shared.llm import LLMConfig, complete_with_retry
from forge.shared.privacy import PrivacyTier

# Contract code may only reach local models — there is no cloud tier and no fallback (the URL
# guard in guards.py says the same about the endpoint). Unconditional, not a setting: the router
# enforces it on every call, so a role that could overflow off-box is refused (403) rather than
# quietly served from the cloud while the guard is looking at the base URL.
CARTOGRAPHER_PRIVACY: PrivacyTier = "local"

# (system, user, max_tokens) -> completion text. Injectable so tests never touch the router.
CompleteFn = Callable[[str, str, int], str]

_FILE_SYSTEM = (
    "You summarize one source file for a repository map used by coding agents. Reply with "
    "4-8 plain sentences: the file's responsibility, its key exported symbols and what they do, "
    "what it depends on, and anything surprising. No headings, no code fences, no praise."
)
_ROLLUP_SYSTEM = (
    "You summarize one module of a repository from its file summaries. Reply with one tight "
    "paragraph (its purpose and shape) followed by up to 6 bullet lines for the load-bearing "
    "files. Plain markdown, no headings."
)
_ARCH_SYSTEM = (
    "You write the architecture overview of a repository from its module summaries. Cover: "
    "what the system does, the major modules and how they depend on each other, where requests/"
    "data enter and leave, and where a newcomer should start reading. Max ~40 lines of plain "
    "markdown; '##' headings allowed."
)
_CONDENSE_SYSTEM = (
    "You condense part of a repository's summaries into an interim digest that a later "
    "synthesis pass will merge with its sibling digests. Keep concrete names (files, modules, "
    "key symbols), responsibilities, and dependencies; drop repetition and filler. Plain "
    "markdown bullet lines, no headings, at most ~30 lines."
)


@dataclass
class SummarizeStats:
    files_total: int = 0
    summarized: int = 0
    cache_hits: int = 0
    llm_calls: int = 0
    rollups_built: int = 0
    architecture_built: bool = False
    skipped: list[str] = field(default_factory=list)  # path: reason
    failed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # call-level exceptions (first line each)


def _default_complete(settings: CartographerSettings, model: str) -> CompleteFn:
    cfg = LLMConfig(
        backend="openai",
        openai_base_url=settings.openai_base_url,
        # The OpenAI SDK refuses an empty key outright; routers that don't check auth accept
        # any placeholder, and ones that do will 401 with a message naming the real problem.
        openai_api_key=settings.openai_api_key or "unset",
        timeout_seconds=settings.llm_timeout_seconds,
        privacy=CARTOGRAPHER_PRIVACY,
    )

    def call(system: str, user: str, max_tokens: int) -> str:
        return complete_with_retry(
            cfg, system=system, user_message=user, model=model, max_tokens=max_tokens
        )

    return call


def _guarded(fn: CompleteFn, label: str, stats: SummarizeStats) -> CompleteFn:
    """One bad call (auth, timeout, router hiccup) becomes a recorded failure, never a crash —
    the entry stays uncached, so the next run retries it."""

    def call(system: str, user: str, max_tokens: int) -> str:
        try:
            return fn(system, user, max_tokens)
        except Exception as exc:  # noqa: BLE001 - resilience boundary, reason is recorded
            stats.errors.append(f"{label}: {type(exc).__name__}: {exc}")
            return ""

    return call


def _packed(sections: list[str], budget: int) -> list[list[str]]:
    """Greedy-pack sections into batches whose joined length stays under ``budget``."""
    batches: list[list[str]] = [[]]
    size = 0
    for section in sections:
        section = section[:budget]  # a single oversized section can't be split further
        if batches[-1] and size + len(section) > budget:
            batches.append([])
            size = 0
        batches[-1].append(section)
        size += len(section)
    return batches


def _fit_to_budget(
    sections: list[str],
    header: str,
    budget: int,
    condense: CompleteFn,
    condense_max_tokens: int,
    stats: SummarizeStats,
    concurrency: int = 1,
) -> list[str] | None:
    """Condense sections (map-reduce, repeated) until one synthesis prompt fits ``budget``.

    Condensing is mechanical compression, so it runs on the sweep tier — high-volume,
    judgment-light, and the sweep seat demonstrably handles budget-sized prompts (the
    synthesis seat 502s on near-context prefills). Returns None when a condense call fails —
    the caller leaves the artifact uncached so the next run retries the whole chain.
    """
    while sum(len(s) for s in sections) + len(header) > budget and len(sections) > 1:
        batches = _packed(sections, budget - len(header))
        stats.llm_calls += len(batches)

        def condense_batch(numbered: tuple[int, list[str]]) -> str:
            i, batch = numbered
            prompt = f"{header} (part {i})\n\n" + "\n\n".join(batch)
            text = condense(_CONDENSE_SYSTEM, prompt, condense_max_tokens)
            if not text.strip():
                # One flaky call (proxy deadline, seat hiccup) must not sink a 25-batch
                # reduction — a 2026-08-28 run lost a whole module rollup to a single 502.
                text = condense(_CONDENSE_SYSTEM, prompt, condense_max_tokens)
            return text

        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            results = list(pool.map(condense_batch, enumerate(batches, start=1)))
        if any(not text.strip() for text in results):
            return None
        condensed = [f"### part {i}\n{text}" for i, text in enumerate(results, start=1)]
        if len(condensed) >= len(sections):  # completions as large as their inputs; give up
            return condensed
        sections = condensed
    return sections


def _file_prompt(entry: FileEntry, content: str, limit: int) -> str:
    parts = [
        f"Path: {entry.path}",
        f"Language: {entry.language}",
        f"Module: {entry.module}",
    ]
    if entry.symbols:
        parts.append("Symbols: " + "; ".join(entry.symbols))
    if entry.imports:
        parts.append("Imports: " + "; ".join(entry.imports))
    parts.append("")
    parts.append(content[:limit])
    return "\n".join(parts)


def load_index(config: MapConfig, target: TargetConfig) -> RepoIndex:
    index_path = config.out_dir(target) / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"no index for target '{target.name}' — run `forge map structure` first"
        )
    return RepoIndex.model_validate_json(index_path.read_text())


def summarize_target(
    config: MapConfig,
    settings: CartographerSettings,
    target: TargetConfig,
    sweep: CompleteFn | None = None,
    synthesis: CompleteFn | None = None,
) -> SummarizeStats:
    index = load_index(config, target)
    mirror = config.mirror_dir(target)
    cache = SummaryCache(config.output_root)
    stats = SummarizeStats(files_total=len(index.files))
    sweep = _guarded(sweep or _default_complete(settings, settings.sweep_model), "sweep", stats)
    synthesis = _guarded(
        synthesis or _default_complete(settings, settings.synthesis_model), "synthesis", stats
    )

    def summarize_file(entry: FileEntry) -> None:
        if entry.skipped:
            stats.skipped.append(f"{entry.path}: {entry.skipped}")
            return
        path = cache.file_summary(entry.blob_sha)
        if SummaryCache.get(path) is not None:
            stats.cache_hits += 1
            return
        source = mirror / entry.path
        try:
            content = source.read_text(errors="replace")
        except OSError:
            # Mirror drifted since the structural pass (file gone) — a stale entry, not a crash.
            stats.skipped.append(f"{entry.path}: missing from mirror (re-run structure)")
            return
        stats.llm_calls += 1
        text = sweep(
            _FILE_SYSTEM,
            _file_prompt(entry, content, settings.max_prompt_chars),
            settings.file_summary_max_tokens,
        )
        if not text.strip():
            stats.failed.append(entry.path)  # uncached → retried on the next run
            return
        SummaryCache.put(path, text)
        stats.summarized += 1

    with ThreadPoolExecutor(max_workers=max(1, config.concurrency)) as pool:
        list(pool.map(summarize_file, index.files))

    # Module rollups (synthesis tier), keyed by the members' blob hashes.
    module_digests: dict[str, str] = {}
    summaries_by_module: dict[str, list[tuple[FileEntry, str]]] = {}
    for entry in index.files:
        text = SummaryCache.get(cache.file_summary(entry.blob_sha))
        if text is not None:
            summaries_by_module.setdefault(entry.module, []).append((entry, text))

    for module in index.modules:
        members = summaries_by_module.get(module.name, [])
        if not members:
            continue
        digest = rollup_digest(module.name, (e.blob_sha for e, _ in members))
        module_digests[module.name] = digest
        if SummaryCache.get(cache.rollup(digest)) is not None:
            continue
        header = f"Module: {module.name} ({module.path})"
        sections = _fit_to_budget(
            [f"### {e.path}\n{text}" for e, text in members],
            header,
            settings.synthesis_prompt_budget_chars,
            sweep,
            settings.condense_max_tokens,
            stats,
            concurrency=config.concurrency,
        )
        if sections is None:
            stats.failed.append(f"rollup:{module.name}")
            continue
        stats.llm_calls += 1
        text = synthesis(
            _ROLLUP_SYSTEM, f"{header}\n\n" + "\n\n".join(sections), settings.rollup_max_tokens
        )
        if text.strip():
            SummaryCache.put(cache.rollup(digest), text)
            stats.rollups_built += 1
        else:
            stats.failed.append(f"rollup:{module.name}")

    # Architecture doc over the rollups.
    arch_digest = None
    if module_digests:
        arch_digest = architecture_digest(module_digests.values())
        rollup_texts = {
            name: SummaryCache.get(cache.rollup(digest))
            for name, digest in sorted(module_digests.items())
        }
        if any(text is None for text in rollup_texts.values()):
            # The arch digest is keyed on member digests, not rollup content — a doc built
            # over a missing rollup would cache permanently and never heal. Fail closed.
            stats.failed.append("architecture")
            arch_digest = None
        elif SummaryCache.get(cache.architecture(arch_digest)) is None:
            header = f"Repository: {index.target.name}"
            sections = _fit_to_budget(
                [f"## {name}\n{text}" for name, text in rollup_texts.items()],
                header,
                settings.synthesis_prompt_budget_chars,
                sweep,
                settings.condense_max_tokens,
                stats,
                concurrency=config.concurrency,
            )
            text = ""
            if sections is not None:
                stats.llm_calls += 1
                text = synthesis(
                    _ARCH_SYSTEM,
                    f"{header}\n\n" + "\n\n".join(sections),
                    settings.architecture_max_tokens,
                )
            if text.strip():
                SummaryCache.put(cache.architecture(arch_digest), text)
                stats.architecture_built = True
            else:
                stats.failed.append("architecture")
                arch_digest = None

    _write_manifest(config, target, index, cache, module_digests, arch_digest)
    return stats


def _write_manifest(
    config: MapConfig,
    target: TargetConfig,
    index: RepoIndex,
    cache: SummaryCache,
    module_digests: dict[str, str],
    arch_digest: str | None,
) -> None:
    """``summaries.json`` — the render stage's lookup table into the cache."""
    out = config.out_dir(target)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "target": target.name,
        "files": {
            e.blob_sha: {
                "path": e.path,
                "module": e.module,
                "summary_file": str(cache.file_summary(e.blob_sha).relative_to(config.output_root)),
                "available": SummaryCache.get(cache.file_summary(e.blob_sha)) is not None,
            }
            for e in index.files
        },
        "modules": {
            name: {
                "digest": digest,
                "rollup_file": str(cache.rollup(digest).relative_to(config.output_root)),
            }
            for name, digest in module_digests.items()
        },
        "architecture": (
            {
                "digest": arch_digest,
                "file": str(cache.architecture(arch_digest).relative_to(config.output_root)),
            }
            if arch_digest
            else None
        ),
    }
    (out / "summaries.json").write_text(json.dumps(manifest, indent=2) + "\n")
