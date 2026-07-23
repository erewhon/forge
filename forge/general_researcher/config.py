from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from forge.shared.envfile import ENV_FILES
from forge.shared.llm import LLMConfig


class GeneralResearcherSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GENERAL_RESEARCHER_", env_file=ENV_FILES, extra="ignore"
    )

    project_dir: Path = Path.home() / "projects" / "research"

    llm_backend: Literal["openai", "anthropic"] = "openai"
    openai_base_url: str = "http://localhost:4000/v1"
    openai_api_key: str = ""
    research_model: str = "research"  # router alias -> self-hosted model (local, private)
    synthesis_model: str = "coder"  # planning/synthesis (also self-hosted)
    anthropic_model: str = "claude-sonnet-4-6"

    max_sprints_per_run: int = 5
    score_threshold: int = 7
    max_findings_tokens: int = 4000

    # Adversarial verification panel: instead of one verifier, fan out these diverse router models
    # (harness consumer #3) — each scores + challenges adversarially, then scores are median-
    # aggregated (robust to a lenient/harsh outlier) and the challenges drive the next sprint. The
    # panel always runs through the router (which hosts them), even when llm_backend="anthropic".
    # Privacy: all three members are self-hosted homelab models (models.yaml backend=vllm, never
    # external), so research findings never leave the lab during verification — the whole research
    # loop is now local end-to-end. Diversity holds across 2+ model families for a real adversarial
    # cross-check: coder=Qwen3.6-35B (hypatia GPU), gptoss=gpt-oss-120B (hekaton CPU, reasoning),
    # m2.7-local=MiniMax-M2.7 (hekaton CPU) — and never the research_model itself (no self-grading).
    # floor=2 leaves headroom for one slow CPU member to time out without dropping the panel.
    # Tradeoff: the hekaton CPU members are markedly slower than the old cloud glm/qwen3.6-plus —
    # accepted for airtight privacy (swapped to self-hosted 2026-07-23).
    verifier_panel_models: list[str] = ["coder", "gptoss", "m2.7-local"]
    verifier_panel_floor: int = 2  # min members that must respond+parse, else degrade

    # Synthesizer ensemble (research panel followup #2): instead of one model writing the final
    # answer, generate a candidate synthesis from each of these models, judge-pick the most
    # coherent, then graft in the unique key_sources / open_questions the runners-up surfaced. Runs
    # through the router. Floor 1 means a single parseable candidate is enough; 0 candidates falls
    # back to a single-model synthesis so the run always produces an answer. Both members are
    # self-hosted (models.yaml backend=vllm, never external), so synthesis stays local too. Kept on
    # the fast GPU Qwen aliases (coder=Qwen3.6-35B hypatia, research=Qwen3-Next-80B archimedes)
    # rather than the slow hekaton-CPU gptoss: synthesis is generation-heavy (long outputs) and the
    # ≥2-family diversity requirement applies to the adversarial verifier, not this generator.
    synthesizer_panel_models: list[str] = ["coder", "research"]
    synthesizer_panel_floor: int = 1

    always_deepen: bool = False

    def llm_cfg(self) -> LLMConfig:
        return LLMConfig(
            backend=self.llm_backend,
            openai_base_url=self.openai_base_url,
            openai_api_key=self.openai_api_key,
            anthropic_model=self.anthropic_model,
        )


settings = GeneralResearcherSettings()
