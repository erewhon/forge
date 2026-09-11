from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from forge.shared.privacy import PrivacyTier


class DependabotSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DEPENDABOT_")

    branch_prefix: str = "deps"
    # "" auto-detects by manifest (uv.lock -> uv, go.mod -> go); "uv"/"go" forces a backend.
    ecosystem: str = ""
    auto_log_path: Path = Path(__file__).parent / "logs" / "auto.jsonl"

    signoff_max_tokens: int = 4096
    signoff_timeout: float = 180.0
    # Which reviewer roster seats the sign-off quorum. Dependency bumps are the routine lane, so
    # the default is the all-local trio (lightning/gpt-oss/coder-next — zero metered tokens);
    # set DEPENDABOT_SIGNOFF_LANE=frontier to restore the sonnet-anchored roster. Decided
    # 2026-08-18 after the 7-run shadow trial: local was at parity on routine diffs; sonnet's
    # unique catches clustered on security-critical code, which bumps route to the supply-chain
    # lens anyway.
    signoff_lane: str = "local"
    # X-Router-Privacy tier for the redundancy-cluster call (`coder` via the router). The
    # sign-off gate takes its tier from the roster it seats (signoff_lane → local/frontier).
    router_privacy: PrivacyTier = "zdr"
    scan_timeout: int = 120
    audit_timeout: int = 300
    metadata_timeout: float = 20.0
    max_candidates: int = 20
    require_attestation: bool = True


settings = DependabotSettings()
