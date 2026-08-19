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

    # Fetch every cited URL and drop the ones that 404. The research model fabricates citations —
    # complete plausible URLs, including one invented ProPublica article manufactured to support a
    # claim that something did NOT happen. "Has sources" does not catch that, because the sources
    # are non-empty, just fake. Existence is decidable without a model, so this runs before a
    # finding is written. Set check_sources_proxy to the router tool proxy's egress so a source the
    # researcher could reach is one this check can reach (otherwise blocked-but-real sources look
    # dead from here). Only 404/410 are dropped; 403/timeouts are treated as unknown, never dead.
    check_sources: bool = True
    check_sources_timeout: float = 15.0
    check_sources_proxy: str | None = None

    # Adversarial verification panel: instead of one verifier, fan out these diverse router models
    # (harness consumer #3) — each scores + challenges adversarially, then scores are median-
    # aggregated (robust to a lenient/harsh outlier) and the challenges drive the next sprint. The
    # panel always runs through the router, even when llm_backend="anthropic". Never the
    # research_model itself (no self-grading).
    #
    # TEMPORARY (2026-07-26): partially de-localised, pending a GPU. The all-self-hosted panel
    # ["coder","gptoss","m2.7-local"] could not actually run — every non-Qwen self-hosted model
    # lives on hekaton, which is CPU-only (4x Xeon E7-4850 v2, no AVX2). Measured on a real sprint,
    # gptoss took 521s and m2.7-local >1600s against the panel's 120s timeout, so all three CPU
    # seats timed out on EVERY verification. What survived was `coder` graded twice — and the median
    # of two values is their mean, so "robust to a lenient/harsh outlier" was one model averaged
    # with itself, while the claim-verification / counter-narrative / actionability lenses went
    # ungraded entirely. Nominally 3-family, actually 1-family, and silent about it.
    #
    # So: keep the local Qwen seat and buy real family diversity from two vetted OpenCode Zen models
    # — glm=GLM-5.2 (Zhipu), kimi=Kimi-K2.7-Code (Moonshot). Per Zen's docs those are zero-retention
    # and not trained on; the free tier (data may train the model) and the OpenAI/Anthropic routes
    # (30-day retention) are excluded, and test_config_privacy enforces both exclusions. Findings DO
    # leave the homelab for verification under this config — an explicit, temporary trade for a
    # panel that functions at all. Revert to self-hosted-only once a non-Qwen model that is NOT the
    # research model can run on GPU.
    #
    # UPDATE (2026-07-31): m3 swapped for kimi. The GPU condition above was met in a twist:
    # MiniMax-M2.7-REAP landed on archimedes — but as the `research`/`thinker` alias, i.e. it IS the
    # research model now, so it is barred from its own panel (no self-grading), and m3=MiniMax-M3
    # would have a MiniMax model grading MiniMax research output (family-correlated leniency, the
    # soft version of the same problem). kimi keeps the seat vetted-Zen and adds a fourth family.
    # The remaining local candidates (ling-flash-local ~80s at 1.9k-tok prompts, gptoss 521s on a
    # real sprint, both hekaton CPU) only fit if the 120s panel timeout is raised; deliberately not
    # taken — revert stays gated on a GPU seat (e.g. gpt-oss on delphi, weights already staged).
    #
    # UPDATE (2026-08-18): coder swapped for gpt-oss — two reasons, one overdue. (1) The GPU gate
    # above was met: gpt-oss-120b now serves on delphi GPU (~49 t/s vs the 521s CPU run that
    # disqualified it; a ~2k-tok verdict fits the 120s panel timeout with room). (2) The router's
    # `research` role flipped to qwen3-coder-next on 2026-08-13, which made the `coder` seat (also
    # Qwen) a family self-grader — the exact soft failure the 07-31 rewire banned; the family map
    # in test_config_privacy had gone stale ("research": "minimax") so the guard test never fired.
    # Lightning (NVIDIA, talos) is deliberately NOT seated: it is the research role's #2 failover,
    # so on an archimedes outage it becomes the research model and would self-grade. Deriving the
    # family maps from models.yaml (so role flips can't silently rot the invariant) is a filed task.
    verifier_panel_models: list[str] = ["gpt-oss", "glm", "kimi"]
    verifier_panel_floor: int = 2  # min members that must respond+parse, else degrade

    # Lane switch (2026-08-18, mirroring the code-review roster_for_lane): "default" = the panel
    # above (one local seat + two vetted-Zen cloud seats); "local" = verifier_panel_models_local —
    # NOTHING leaves the homelab. In the local lane family purity is a deliberate, accepted trade
    # for privacy (Steven's call): `coder` shares the research model's Qwen family and `lightning`
    # is the research role's #2 failover, but the alternative is shipping findings to the cloud.
    # The no-self-grading tests pin the DEFAULT panel only; the local lane pins full self-hosting
    # instead. Unknown lane values fall through to the default panel (the vetted one).
    # Env: GENERAL_RESEARCHER_PANEL_LANE=local, or the CLI's --local flag.
    panel_lane: str = "default"
    verifier_panel_models_local: list[str] = ["gpt-oss", "lightning", "coder"]

    def active_verifier_panel(self) -> list[str]:
        if self.panel_lane == "local":
            return self.verifier_panel_models_local
        return self.verifier_panel_models

    # Synthesizer ensemble (research panel followup #2): instead of one model writing the final
    # answer, generate a candidate synthesis from each of these models, judge-pick the most
    # coherent, then graft in the unique key_sources / open_questions the runners-up surfaced. Runs
    # through the router. Floor 1 means a single parseable candidate is enough; 0 candidates falls
    # back to a single-model synthesis so the run always produces an answer. Both members are
    # self-hosted (models.yaml backend=vllm, never external) — synthesis stays local *deliberately*,
    # even while the verifier panel is temporarily on Zen: these are fast GPU aliases
    # (coder=Qwen3.6-35B hypatia; research=MiniMax-M2.7-REAP archimedes since 2026-07-31, formerly
    # Qwen3-Next-80B) with no latency problem to solve, so there is nothing to buy by sending
    # findings off-box here. Keeping them local bounds the exposure to the verification step alone.
    # The ≥2-family diversity requirement applies to the adversarial verifier, not this generator —
    # though the pair happens to span two families (Qwen + MiniMax) since the alias swap.
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
