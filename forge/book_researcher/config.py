from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from forge.shared.envfile import ENV_FILES
from forge.shared.llm import LLMConfig
from forge.shared.privacy import PrivacyTier, tier_for_lane


class BookResearcherSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BOOK_RESEARCHER_", env_file=ENV_FILES, extra="ignore"
    )

    # Project paths
    project_dir: Path = Path.home() / "projects" / "book-research"

    # AI backend: "anthropic" or "openai" (for local/router endpoints)
    llm_backend: Literal["openai", "anthropic"] = "openai"

    # OpenAI-compatible models (used when llm_backend == "openai")
    openai_base_url: str = "http://localhost:4000/v1"
    openai_api_key: str = ""
    research_model: str = "research"  # router alias -> self-hosted model (local, private)
    synthesis_model: str = "coder"  # planning, verification, synthesis (also self-hosted)

    # Anthropic models (fallback when llm_backend == "anthropic")
    anthropic_model: str = "claude-sonnet-4-6"

    # Sprint settings
    max_sprints_per_run: int = 3
    score_threshold: int = 7  # minimum score (1-10) to accept findings
    max_findings_tokens: int = 4000  # truncate findings context for verifier

    # Fetch every cited URL and drop the ones that 404 — the research model fabricates citations,
    # and "has sources" does not catch it because the sources are non-empty, just fake. Existence
    # is decidable without a model. Set check_sources_proxy to the router tool proxy's egress so a
    # source the researcher could reach is one this check can reach. Only 404/410 are dropped;
    # 403s and timeouts are unknown, never dead. See forge/shared/source_check.py.
    check_sources: bool = True
    check_sources_timeout: float = 15.0
    check_sources_proxy: str | None = None

    # Adversarial verification panel (ensemble harness consumer #3): instead of one verifier, fan
    # out these diverse router models — each scores + challenges adversarially; scores are median-
    # aggregated (robust to a lenient/harsh outlier) and the challenges drive the next sprint. The
    # panel always runs through the router, even when llm_backend="anthropic".
    #
    # TEMPORARY (2026-07-26): partially de-localised, pending a GPU. The all-self-hosted panel
    # ["coder","gptoss","m2.7-local"] could not actually run — every non-Qwen self-hosted model
    # lives on hekaton, which is CPU-only (4x Xeon E7-4850 v2, no AVX2). Measured on a real sprint,
    # gptoss took 521s and m2.7-local >1600s against the panel's 120s timeout, so all three CPU
    # seats timed out on EVERY verification and only `coder` (graded twice) survived — nominally
    # 3-family, actually 1-family, with three lenses ungraded and no complaint in the output.
    #
    # So: keep the local Qwen seat and buy real family diversity from two vetted OpenCode Zen models
    # — glm=GLM-5.2 (Zhipu), kimi=Kimi-K2.7-Code (Moonshot). Per Zen's docs those are zero-retention
    # and not trained on; the free tier (data may train the model) and the OpenAI/Anthropic routes
    # (30-day retention) are excluded, and test_config_privacy enforces both exclusions. Book
    # findings DO leave the homelab for verification under this config — an explicit, temporary
    # trade for a panel that functions.
    #
    # UPDATE (2026-07-31): m3 swapped for kimi — the `research` alias is now MiniMax-M2.7-REAP on
    # archimedes, so any MiniMax panel seat would grade its own family's research output. See the
    # general researcher's config for the full rationale; revert to self-hosted-only stays gated on
    # a GPU seat for a non-Qwen model that is not the research model.
    #
    # UPDATE (2026-08-18): coder swapped for gpt-oss. The GPU gate was met (gpt-oss-120b on delphi
    # GPU, ~49 t/s), and the `research` role's 2026-08-13 flip to qwen3-coder-next had quietly made
    # the Qwen `coder` seat a family self-grader while the stale family map in test_config_privacy
    # kept the guard test green. Lightning stays out: it is research's #2 failover and can BE the
    # research model. Full rationale in the general researcher's config.
    verifier_panel_models: list[str] = ["gpt-oss", "glm", "kimi"]
    verifier_panel_floor: int = 2  # min members that must respond+parse, else degrade

    # Lane switch (2026-08-18): "local" swaps the panel to verifier_panel_models_local — nothing
    # leaves the homelab; family purity deliberately traded for privacy (see the general
    # researcher's config for the full rationale). Env: BOOK_RESEARCHER_PANEL_LANE=local, or the
    # CLI's --local flag. Unknown lane values fall through to the default (vetted) panel.
    panel_lane: str = "default"
    verifier_panel_models_local: list[str] = ["gpt-oss", "lightning", "coder"]

    # X-Router-Privacy (2026-09-11): the lane also picks the tier every router call in the run
    # sends — research model, planner, panel, outline pool. See the general researcher's config
    # for the full rationale; in short the router enforces the lane on the wire (local: no
    # overflow off-box, cloud aliases refused; default: vetted seats served from a
    # zero-retention-enforceable endpoint, Zen chain members skipped). `panel_privacy` overrides
    # the lane-derived tier (BOOK_RESEARCHER_PANEL_PRIVACY / --privacy).
    panel_privacy: PrivacyTier | None = None

    def active_verifier_panel(self) -> list[str]:
        if self.panel_lane == "local":
            return self.verifier_panel_models_local
        return self.verifier_panel_models

    def privacy_tier(self) -> PrivacyTier:
        """The X-Router-Privacy tier for this run: the explicit override, else lane-derived
        (local → ``local``, anything else → ``zdr``)."""
        return self.panel_privacy or tier_for_lane(self.panel_lane)

    # Outline lifecycle (`forge book lint --critic` / `revise` / `decompose`): an ordered failover
    # pool, tried first-to-last. Every outline call is evidence-in / structured-out (the schema is
    # the validator). Same vetting rules as the panel (test_config_privacy).
    #
    # gemma4 first (2026-08-30): on the Qualeval v2 board the outline work's dimensions —
    # instruction, research, adversarial, code_review-as-critique — put gemma4-26b at 0.92 vs
    # coder (qwen3.6-hypatia) at 0.77, the weakest seated model on exactly those axes (its 0.85
    # composite rides on tool_use/codegen, which the outline never exercises). Measured on the
    # corruption book's 12 reviews: gemma4's revise proposal was better targeted (caption rule,
    # verify-or-retract questions) at ~10 min per call vs coder's 30 s. A few calls a month per
    # book, so quality wins; coder is the fast failover. lightning (0.87 fit, 97 t/s) truncated
    # its JSON under the 16k budget both attempts — investigate before seating it here.
    outline_models: list[str] = ["gemma4", "coder"]
    outline_timeout: float = 300.0
    outline_max_tokens: int = 16384
    # Reviews per chapter fed to `revise` (most recent first) — bounds the evidence prompt.
    max_reviews_per_chapter: int = 4

    @property
    def sprints_dir(self) -> Path:
        return self.project_dir / "sprints"

    @property
    def knowledge_dir(self) -> Path:
        return self.project_dir / "knowledge"

    @property
    def outline_file(self) -> Path:
        return self.project_dir / "outline.yaml"

    @property
    def framing_file(self) -> Path:
        return self.project_dir / "outline-framing.json"

    def llm_cfg(self) -> LLMConfig:
        if self.llm_backend == "anthropic" and self.privacy_tier() != "any":
            # The native Anthropic backend bypasses the router: nothing can hold it to a tier.
            raise ValueError(
                f"llm_backend='anthropic' cannot honour privacy tier {self.privacy_tier()!r} "
                f"(lane {self.panel_lane!r}); use the router backend, or set "
                "BOOK_RESEARCHER_PANEL_PRIVACY=any to state the trade explicitly"
            )
        return LLMConfig(
            backend=self.llm_backend,
            openai_base_url=self.openai_base_url,
            openai_api_key=self.openai_api_key,
            anthropic_model=self.anthropic_model,
            privacy=self.privacy_tier(),
        )


settings = BookResearcherSettings()
