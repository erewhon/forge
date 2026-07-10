from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from agents.shared.llm import LLMConfig


class CodingPipelineSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CODING_PIPELINE_")

    # Per-epic run dirs (framing, tree, wave records, journal) live here.
    runs_dir: Path = Path.home() / "Projects" / "erewhon" / "meta" / "pipeline-runs"

    # Loop bounds. max_waves is per *run* (re-run to continue, like the research harness's
    # --max-sprints); max_leaf_attempts is per leaf across the whole epic — at the cap the leaf
    # escalates to a human instead of retrying blind.
    max_waves: int = 3
    wave_size: int = 4
    max_leaf_attempts: int = 2

    # Leaves in flight per wave. 1 = the serial path (no workspaces, dx sandbox). Above 1,
    # leaves fan out into per-leaf jj workspaces under run-once sandboxes and integrate
    # through the serial reconcile barrier. Default 3 since the deliberate-conflict smoke
    # passed (2026-07-07: colliding pair detected, one landed, one demoted-and-retried
    # onto the updated head); wave_size is the outer bound either way.
    dispatch_concurrency: int = 3

    # All epic work lands on {branch_prefix}/{epic_slug}; main only moves at the epic gate.
    branch_prefix: str = "pipeline"

    # A0 inventory caps — the architect prompt budget. Rendered inventory.md is trimmed (tree
    # first) to fit inventory_max_chars; per-section drops are counted, never silent.
    inventory_max_chars: int = 40_000
    inventory_tree_depth: int = 3

    # Architect LLM (strong tier) — the headless path only. The interactive Fable session IS the
    # architect during the free window and never touches these.
    llm_backend: Literal["openai", "anthropic"] = "openai"
    openai_base_url: str = "http://localhost:4010/v1"
    openai_api_key: str = ""
    architect_model: str = "coder"
    anthropic_model: str = "claude-sonnet-4-6"
    architect_max_tokens: int = 8192
    architect_timeout: float = 240.0
    decompose_max_tokens: int = 16_000  # trees are big: N leaves x full worker specs
    default_auto_max_files: int = 5  # every Auto-* leaf gets a max_files cap, no exceptions
    # Floor for autonomous leaves whose tier is unset/"auto": the router's bare "auto"
    # often returns text-only (zero tool calls) through opencode (e2e dry-run), so
    # Auto-* leaves get a tool-capable tier. Explicit auto-free/auto-full stand.
    leaf_model_tier: str = "coder"

    # Wave verification (advisory review pass over the wave diff)
    review_max_tokens: int = 4096
    review_timeout: float = 180.0
    # Epic-gate seat budget. Thinking models spend reasoning tokens INSIDE max_tokens: at
    # 4096 the anthropic seat (sonnet-5 via the router) burned the whole budget on a
    # 140k-char epic diff and returned ok + ZERO text — "no verdict", gate blocked
    # structurally (live finding, 2026-07-07). 16k leaves room to think AND answer.
    epic_gate_signoff_max_tokens: int = 16_000
    review_max_findings: int = 12  # cap the candidate pool before the confirm vote
    confirm_concurrency: int = 4

    # Epic gate size guard. A diff at or under epic_gate_max_diff_chars is signed off in one
    # call per seat; a larger one goes map-reduce — deterministic per-file split, one gatekeeper
    # summary per slice, then the full-quorum verdict over the slice summaries. A gate must
    # never judge code it hasn't seen, so there is no silent slice drop: a failed slice summary
    # fails the gate closed, and a diff splitting past epic_gate_max_chunks blocks instead of
    # truncating (gate sub-epics separately, or raise the cap deliberately).
    epic_gate_max_diff_chars: int = 300_000
    epic_gate_chunk_chars: int = 100_000
    # Map summaries are mechanical, but a thinking seat serving a slice via failover
    # spends reasoning tokens inside this budget too — and a failed slice fails the
    # gate closed. Same hazard as epic_gate_signoff_max_tokens, smaller scale.
    epic_gate_map_max_tokens: int = 8192
    epic_gate_map_concurrency: int = 4
    epic_gate_max_chunks: int = 64
    # Slice summaries are mechanical, so the map pool tries this seat first (failover to the
    # rest) instead of spending metered tokens on every slice. The reduce verdict is still the
    # full cross-family quorum — every seat, unanimous.
    epic_gate_map_preferred: str = "local"

    def llm_cfg(self) -> LLMConfig:
        return LLMConfig(
            backend=self.llm_backend,
            openai_base_url=self.openai_base_url,
            openai_api_key=self.openai_api_key,
            anthropic_model=self.anthropic_model,
        )


settings = CodingPipelineSettings()
