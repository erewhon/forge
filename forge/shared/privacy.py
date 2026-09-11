"""``X-Router-Privacy``: the privacy tier every forge call to the LLM router carries.

The router (llm-router-go, 2026-09-10) constrains, per request, where a prompt may go:

* ``local`` — fleet hardware only; nothing leaves the building. Role candidates, chain members,
  and overflow entries that are not node-pinned are filtered out of the walk.
* ``zdr`` — local seats, plus cloud endpoints the router can hold to zero data retention on the
  wire (OpenRouter today: ``provider: {"zdr": true}`` is attached and OpenRouter refuses when no
  endpoint qualifies). OpenCode Zen is deliberately NOT zdr-enforceable — its policy covers the
  upstream providers, not transit through OpenCode's own servers — so a ``zdr`` request for a
  chain like ``glm`` is served from the OpenRouter member and the Zen member is skipped.
* ``any`` — no requirement of the caller's own. This is also what an ABSENT header means, which is
  why forge never leaves it absent: ``ApiExecutor`` and ``LLMConfig`` both require a tier, so a
  call site has to say what it wants rather than inherit the permissive default by omission.

A caller can tighten a role's declared tolerance, never loosen it: the roles forge uses
(``research``/``coder``/``thinker``) declare ``locality: local_or_zdr``, so they get the ZDR
directive even under ``any``. The header earns its keep on roles for ``local``, and on directly
named aliases (``glm``, ``kimi``, ``sonnet`` …) for both tiers — a bare name has no role contract
behind it, so the header is the only thing that can constrain it.

Refusals are a distinct outcome, not an outage: the router answers a request whose tier excludes
every candidate with **403** and a body carrying ``code: privacy_tier_unavailable`` (plus the
per-seat ``excluded_candidates``), never a 503. A 403 already classifies TERMINAL in the ensemble
harness (fail over, never retry); :func:`privacy_refusal` recognises the body so the seat's
absence is reported as "refused by privacy policy" rather than a generic 403.
"""

from __future__ import annotations

from typing import Any, Literal, get_args

PrivacyTier = Literal["local", "zdr", "any"]
PRIVACY_TIERS: tuple[str, ...] = get_args(PrivacyTier)

PRIVACY_HEADER = "X-Router-Privacy"
# Response headers the router sets on a role/chain-resolved request — captured into
# ExecResult.meta so a run can say which concrete model served a seat and under which tier.
ROUTER_RESOLVED_HEADER = "X-Router-Resolved"
ROUTER_ROLE_HEADER = "X-Router-Role"
ROUTER_OVERFLOW_HEADER = "X-Router-Overflow"

PRIVACY_REFUSED_CODE = "privacy_tier_unavailable"
PRIVACY_REFUSED_TYPE = "privacy_policy_violation"


def check_tier(tier: str) -> PrivacyTier:
    """Validate a tier string. Raises ``ValueError`` on anything but the three known values —
    the router refuses an unrecognised value with a 403 on every seat, so catching a typo here
    saves a whole run from being refused at its first call."""
    if tier not in PRIVACY_TIERS:
        raise ValueError(f"unknown privacy tier {tier!r}; want one of {', '.join(PRIVACY_TIERS)}")
    return tier  # type: ignore[return-value]


def privacy_headers(tier: str) -> dict[str, str]:
    """The request header dict for ``tier`` — what goes in an OpenAI client's default_headers."""
    return {PRIVACY_HEADER: check_tier(tier)}


def tier_for_lane(lane: str) -> PrivacyTier:
    """The researchers' lane → tier mapping: the ``local`` lane is the whole point of ``local``;
    every other lane is the vetted default panel, which is exactly "zero retention" — so ``zdr``,
    never ``any``. Unknown lane values fall through to the default panel in the configs, and to
    ``zdr`` here, for the same reason: the stricter tier is the fail-closed one."""
    return "local" if lane == "local" else "zdr"


def router_meta(headers: Any) -> dict[str, str]:
    """Pick the router's routing/privacy response headers out of an httpx-style headers mapping.
    Empty when none are present (a directly named local model sets no role headers)."""
    if headers is None:
        return {}
    out: dict[str, str] = {}
    for key, name in (
        (PRIVACY_HEADER, "router_privacy"),
        (ROUTER_RESOLVED_HEADER, "router_resolved"),
        (ROUTER_ROLE_HEADER, "router_role"),
        (ROUTER_OVERFLOW_HEADER, "router_overflow"),
    ):
        try:
            value = headers.get(key)
        except Exception:  # noqa: BLE001 — a foreign headers object is not worth a crash
            value = None
        if value:
            out[name] = str(value)
    return out


def _error_object(exc: BaseException) -> dict[str, Any] | None:
    """The router's ``error`` object from an OpenAI-SDK status error, whichever shape the SDK
    handed over: the inner error dict (openai-python unwraps ``{"error": {...}}`` for
    ``APIStatusError.body``) or the full envelope."""
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None
    inner = body.get("error")
    if isinstance(inner, dict):
        return inner
    return body


def privacy_refusal(exc: BaseException) -> str | None:
    """If ``exc`` is the router's privacy refusal, a human-readable reason; else ``None``.

    Matched on the structured ``code``/``type`` first, so the reason survives any rewording of
    the message; the message text is a fallback for a proxy that flattened the body."""
    err = _error_object(exc)
    if err is not None and (
        err.get("code") == PRIVACY_REFUSED_CODE or err.get("type") == PRIVACY_REFUSED_TYPE
    ):
        message = str(err.get("message") or "no candidate satisfies the requested tier")
        tier = err.get("privacy_tier")
        excluded = err.get("excluded_candidates")
        reason = "refused by privacy policy"
        if tier:
            reason += f" ({PRIVACY_HEADER}: {tier})"
        reason += f": {message}"
        if isinstance(excluded, list) and excluded:
            reason += " [excluded: " + "; ".join(str(x) for x in excluded) + "]"
        return reason
    text = str(exc)
    if PRIVACY_REFUSED_CODE in text or "privacy tier" in text:
        return f"refused by privacy policy: {text}"
    return None
