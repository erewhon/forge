"""Generate the actual test code for confirmed coverage gaps, and apply it to the working copy.

This is the one *generative* step in the auto-merge loop — the rest of the ensemble only analyzes.
An LLM (a failover pool over ``gen_models``) is handed the SOURCE + EXISTING TESTS context and the
confirmed gaps, and returns structured per-gap test edits. Application is defensive: only paths that
look like test files and stay inside the repo are written, and existing files are appended to rather
than clobbered. The shared tests-only classifier re-checks the working copy afterwards, so a
misbehaving model is caught by the gate even if it slips past application.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from agents.shared.automerge import is_test_path
from agents.shared.ensemble import ApiExecutor, Pool
from agents.shared.panel import structured
from agents.testing_ensemble.config import settings
from agents.testing_ensemble.models import ScoredGap

GEN_SYSTEM = """You write focused, correct automated tests for an existing codebase.

You are given the SOURCE UNDER TEST, its EXISTING TESTS, and a list of confirmed coverage GAPS.
For each gap you address, write test code that closes it.

Hard rules:
- Write ONLY test files. Never modify source, config, CI, or dependency manifests
  (pyproject.toml, package.json, Cargo.toml, go.mod, lockfiles, etc.).
- Add NO new external/third-party dependencies. Use only what the existing tests already import.
- Match the existing tests' framework, style, imports, and file layout exactly.
- Prefer appending to the most relevant existing test file; create a new one only if none fits.
- Tests must be deterministic — no real network, wall-clock, randomness, or sleeps.
- Do not weaken, delete, rename, or skip existing tests.

Return ONLY a JSON object of this shape:
{"tests": [{"gap_target": "<echo the gap target>", "test_file": "<repo-relative path>",
"mode": "append" | "create", "code": "<the test code>", "notes": "<optional>"}]}

`test_file` must be a test-file path (under a tests/ dir, or named test_*, *_test, or *.test.*).
For "append", `code` is a self-contained snippet including any imports it needs. For "create",
`code` is the full file content."""


class GeneratedTest(BaseModel):
    """One test edit the generator proposes for a specific gap."""

    gap_target: str
    test_file: str
    mode: Literal["create", "append"] = "append"
    code: str = ""
    notes: str = ""


class GeneratedTestsEnvelope(BaseModel):
    tests: list[GeneratedTest] = Field(default_factory=list)


def _gap_block(g: ScoredGap) -> str:
    gap = g.gap
    return (
        f"- target: {gap.target}\n"
        f"  type: {gap.gap_type or 'coverage'}\n"
        f"  why: {gap.why_it_matters}\n"
        f"  suggested: {gap.suggested_test}"
    )


def build_gen_user(context: str, gaps: list[ScoredGap]) -> str:
    blocks = "\n".join(_gap_block(g) for g in gaps)
    return f"{context}\n\n## CONFIRMED GAPS TO ADDRESS\n\n{blocks}\n"


def generate_tests(
    context: str, gaps: list[ScoredGap], *, log: Callable[[str], None] | None = None
) -> GeneratedTestsEnvelope:
    """Ask the generator pool for test code closing ``gaps`` (empty envelope if none produced)."""
    pool = Pool(
        role="test-gen",
        executors=[
            ApiExecutor(
                label=f"router:{m}",
                kind="openai",
                model=m,
                base_url=settings.openai_base_url,
                api_key=settings.openai_api_key,
            )
            for m in settings.gen_models
        ],
    )
    res = structured(
        pool=pool,
        schema=GeneratedTestsEnvelope,
        system=GEN_SYSTEM,
        user=build_gen_user(context, gaps),
        max_tokens=settings.gen_max_tokens,
        timeout=settings.per_call_timeout,
    )
    if res.value is None and log:
        log(f"generation produced no usable output: {res.error}")
    return res.value or GeneratedTestsEnvelope()


def apply_generated(
    repo_path: Path, env: GeneratedTestsEnvelope, *, log: Callable[[str], None] | None = None
) -> list[str]:
    """Write the generated tests into ``repo_path``, returning the repo-relative paths written.

    Defensive: a proposed path that isn't a test file, or that escapes the repo, is skipped (the
    gate would reject the whole change anyway, but we never write such a file). Existing files are
    appended to; ``create`` on an existing file degrades to append so nothing is clobbered.
    """
    root = repo_path.resolve()
    written: list[str] = []
    for t in env.tests:
        rel = t.test_file.strip().lstrip("/")
        if not rel or not is_test_path(rel):
            if log:
                log(f"skip non-test path from generator: {t.test_file!r}")
            continue
        target = (root / rel).resolve()
        if not target.is_relative_to(root):
            if log:
                log(f"skip path escaping repo: {t.test_file!r}")
            continue
        if not t.code.strip():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if t.mode == "create" and not target.exists():
            target.write_text(t.code.rstrip("\n") + "\n", encoding="utf-8")
        else:
            existing = target.read_text(encoding="utf-8") if target.exists() else ""
            prefix = "" if not existing else ("\n" if existing.endswith("\n") else "\n\n")
            with target.open("a", encoding="utf-8") as fh:
                fh.write(prefix + t.code.rstrip("\n") + "\n")
        written.append(str(target.relative_to(root)))
    return written
