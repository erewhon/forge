"""Prompts for the refactoring ensemble: smell-finding angles, consolidator, and skeptic lenses.

Finders read the *provided* code in the user message (executors have no file tools); each finder's
system prompt aims it at one class of code smell. The two skeptic lenses are the load-bearing guard
against noise: "safety" kills anything that changes behavior, "worth-it" kills bikeshedding.
"""

from __future__ import annotations

# Each finder hunts ONE class of smell hardest (still reporting anything it sees). The union across
# angles surfaces more than any single reviewer would, before dedup collapses the overlap.
FINDER_ANGLES: tuple[tuple[str, str], ...] = (
    (
        "duplication",
        "repeated logic and copy-paste that should be unified; near-duplicate functions or blocks",
    ),
    (
        "complexity",
        "long functions, deep nesting, high branching, tangled control flow that should be "
        "decomposed",
    ),
    (
        "naming-clarity",
        "misleading or vague names, unclear intent, and comments that compensate for code that "
        "should be clearer",
    ),
    (
        "coupling",
        "leaky abstractions, tight coupling, hidden shared state, and modules that know too much "
        "about each other",
    ),
    (
        "dead-code",
        "unused functions, parameters, and branches; unreachable code; obsolete compat shims",
    ),
    (
        "idiom",
        "non-idiomatic patterns, reinvented standard-library functionality, and language "
        "best-practice violations",
    ),
)

# Adversarial verification lenses — the two guards that keep the plan high-signal.
SKEPTIC_LENSES: tuple[tuple[str, str], ...] = (
    (
        "safety",
        "Would this refactor change observable behavior or break a public API/contract? Read the "
        "code and confirm equivalence. If it is not behavior-preserving, mark real=false.",
    ),
    (
        "worth-it",
        "Is this a material improvement, or bikeshedding / churn for its own sake? Trivial style "
        "nits, and changes whose risk or effort outweighs the benefit, are not worth it. "
        "Mark those real=false.",
    ),
)

_SMELL_SHAPE = (
    'Return ONLY valid JSON: {"smells": [{"location": "file::function or area", '
    '"smell_type": "duplication|complexity|naming|coupling|dead-code|idiom", '
    '"proposed_refactor": "the concrete change", "benefit": "why it helps", '
    '"risk": "behavior-change or API risk, or none", "effort": "small|medium|large", '
    '"impact": "high|medium|low"}]}. Return an empty smells list if the code is already clean.'
)


def finder_system(focus: str, angle_directive: str) -> str:
    return (
        "You are a refactoring reviewer of the code provided by the user. "
        f"FOCUS: {focus}. YOUR ANGLE — hunt hardest for: {angle_directive}. "
        "Propose only BEHAVIOR-PRESERVING refactors that are a material improvement. Never propose "
        "changes that alter behavior or break a public API, and do not bikeshed trivial style. Be "
        "concrete: name the location and the change. " + _SMELL_SHAPE
    )


DEDUP_SYSTEM = (
    "You consolidate refactoring suggestions from several independent reviewers of the SAME code. "
    "Merge suggestions targeting the same code or smell into ONE canonical suggestion; keep "
    "genuinely distinct ones separate. For each canonical suggestion assign a stable id (RF-01, "
    "RF-02, ...), keep the clearest location/proposed_refactor/benefit, take the HIGHEST impact "
    "and the most cautious risk note among the merged duplicates, and list the merged source "
    "locations in merged_from. "
    'Return ONLY valid JSON: {"smells": [{"id": "RF-01", "location": "...", "smell_type": "...", '
    '"proposed_refactor": "...", "benefit": "...", "risk": "...", '
    '"effort": "small|medium|large", "impact": "high|medium|low", "merged_from": ["..."]}], '
    '"dropped": <how many raw suggestions were merged away>}.'
)


def build_dedup_user(raw_json: str) -> str:
    return f"Here are the raw refactoring suggestions as JSON:\n\n{raw_json}\n\nMerge them."


SKEPTIC_BASE = (
    "You are an ADVERSARIAL verifier of a claimed refactoring suggestion. Decide whether it is "
    "REAL: a behavior-preserving, materially worthwhile improvement. Default to real=false. "
    "Reject anything that changes behavior, breaks an API, or is mere style churn. "
    'Return ONLY valid JSON: {"real": true|false, "confidence": "high|medium|low", '
    '"impact": "high|medium|low", "reasoning": "one sentence citing the code"}.'
)


def verify_user(code_context: str, smell_json: str) -> str:
    return (
        f"CODE:\n\n{code_context}\n\n"
        f"CLAIMED REFACTORING (JSON):\n{smell_json}\n\n"
        "Is this a safe, worthwhile, behavior-preserving refactor? Return your verdict as JSON now."
    )
