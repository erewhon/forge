"""forge.shared.privacy: the tier vocabulary, the lane mapping, and refusal recognition."""

from __future__ import annotations

import pytest

from forge.shared.privacy import (
    PRIVACY_HEADER,
    check_tier,
    privacy_headers,
    privacy_refusal,
    router_meta,
    tier_for_lane,
)


@pytest.mark.parametrize("tier", ["local", "zdr", "any"])
def test_known_tiers_pass(tier: str) -> None:
    assert check_tier(tier) == tier
    assert privacy_headers(tier) == {PRIVACY_HEADER: tier}


@pytest.mark.parametrize("bad", ["", "locl", "LOCAL", "eu-only", "unrestricted"])
def test_unknown_tiers_are_refused(bad: str) -> None:
    # The router refuses these with a 403 on every seat; forge refuses them before the run.
    with pytest.raises(ValueError):
        check_tier(bad)


def test_lane_mapping_is_fail_closed() -> None:
    """local → local; the vetted default is exactly 'zero retention', so zdr — never any; an
    unknown lane gets the stricter tier for the same reason the configs give it the vetted
    panel."""
    assert tier_for_lane("local") == "local"
    assert tier_for_lane("default") == "zdr"
    assert tier_for_lane("bogus") == "zdr"


class _Exc(Exception):
    def __init__(self, body, message="err"):
        super().__init__(message)
        self.body = body


def test_refusal_recognised_by_code() -> None:
    reason = privacy_refusal(
        _Exc({"code": "privacy_tier_unavailable", "message": "no local candidate"})
    )
    assert reason == "refused by privacy policy: no local candidate"


def test_refusal_recognised_by_type_and_carries_tier_and_exclusions() -> None:
    reason = privacy_refusal(
        _Exc(
            {
                "type": "privacy_policy_violation",
                "message": "m",
                "privacy_tier": "zdr",
                "excluded_candidates": ["zen/glm-5.2: neither local nor ZDR-enforceable"],
            }
        )
    )
    assert reason is not None
    assert reason.startswith("refused by privacy policy (X-Router-Privacy: zdr): m")
    assert "zen/glm-5.2" in reason


def test_refusal_recognised_inside_an_error_envelope() -> None:
    # Some proxies hand the SDK the full {"error": {...}} body rather than the inner object.
    reason = privacy_refusal(_Exc({"error": {"code": "privacy_tier_unavailable", "message": "x"}}))
    assert reason == "refused by privacy policy: x"


def test_refusal_falls_back_to_message_text() -> None:
    reason = privacy_refusal(
        _Exc("not json", 'role "coder" has no candidate that satisfies privacy tier "local"')
    )
    assert reason is not None and reason.startswith("refused by privacy policy")


def test_non_refusals_are_none() -> None:
    assert privacy_refusal(_Exc({"code": "model_not_found", "message": "nope"})) is None
    assert privacy_refusal(ValueError("boom")) is None


def test_router_meta_picks_only_the_router_headers() -> None:
    headers = {
        "content-type": "application/json",
        "X-Router-Privacy": "local",
        "X-Router-Resolved": "qwen3.6-hypatia",
        "X-Router-Role": "coder",
        "X-Router-Overflow": "",
    }
    assert router_meta(headers) == {
        "router_privacy": "local",
        "router_resolved": "qwen3.6-hypatia",
        "router_role": "coder",
    }
    assert router_meta(None) == {}
    assert router_meta({}) == {}
