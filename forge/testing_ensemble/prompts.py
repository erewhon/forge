"""Prompts for the testing-review ensemble: gap-finding angles, consolidator, and skeptic lenses.

Finders are handed the SOURCE and its EXISTING TESTS in the user message (executors have no file
tools); each finder's system prompt aims it at one class of coverage gap. Skeptic lenses give the
verification panel orthogonal ways to refute a claimed gap — chiefly "is it already covered?".
"""

from __future__ import annotations

# Each finder hunts ONE class of coverage gap hardest (still reporting anything it sees). The union
# across angles surfaces more holes than any single reviewer would, before dedup collapses overlap.
FINDER_ANGLES: tuple[tuple[str, str], ...] = (
    (
        "coverage",
        "untested public functions, branches, and return values — the obvious holes a reader would "
        "expect a test for",
    ),
    (
        "error-paths",
        "exception and failure paths: do raised errors, timeouts, and rejected/invalid inputs have "
        "tests, or only the happy path?",
    ),
    (
        "concurrency",
        "race conditions, ordering assumptions, shared mutable state, and async paths that "
        "lack any guarding test",
    ),
    (
        "edge-cases",
        "empty / None / zero, boundaries, unicode, very large inputs, off-by-one — the untested "
        "corners",
    ),
    (
        "regression-risk",
        "fragile or intricate logic with no guarding test, and non-deterministic behavior "
        "that could regress silently",
    ),
)

# Adversarial verification lenses — each skeptic tries to REFUTE the claimed gap from a different
# angle. The first is the load-bearing one: is the case actually already covered?
SKEPTIC_LENSES: tuple[tuple[str, str], ...] = (
    (
        "coverage-truth",
        "Read the EXISTING TESTS carefully. If any test already exercises this case — directly or "
        "indirectly — the gap is NOT real. Be strict about what is genuinely uncovered.",
    ),
    (
        "testability",
        "Is closing this gap worthwhile, and is a correct, DETERMINISTIC test actually "
        "feasible? If the suggested test is vacuous, flaky, or tests the wrong thing, it is "
        "not a useful gap.",
    ),
)

_GAP_SHAPE = (
    'Return ONLY valid JSON: {"gaps": [{"target": "file::function or behavior", '
    '"gap_type": "coverage|error-path|concurrency|edge-case|regression", '
    '"why_it_matters": "the bug that could slip through untested", '
    '"suggested_test": "a concrete test to add", "severity": "critical|high|medium|low"}]}. '
    "Return an empty gaps list if the existing tests are thorough."
)


def finder_system(focus: str, angle_directive: str) -> str:
    return (
        "You are a test reviewer. You are given SOURCE code and its EXISTING TESTS. Find GAPS: "
        "important behaviors of the source that the existing tests do NOT cover. "
        f"FOCUS: {focus}. YOUR ANGLE — hunt hardest for: {angle_directive}. "
        "Do NOT propose tests for cases the existing tests already cover. Be concrete: name the "
        "function and the specific case. Do not invent gaps to fill a quota. " + _GAP_SHAPE
    )


DEDUP_SYSTEM = (
    "You consolidate test-coverage gaps from several independent reviewers of the SAME code. "
    "Merge gaps that describe the same untested behavior into ONE canonical gap; keep genuinely "
    "distinct gaps separate. For each canonical gap assign a stable id (TG-01, TG-02, ...), keep "
    "the clearest target/why_it_matters/suggested_test, take the HIGHEST severity among the "
    "merged duplicates, and list the merged source targets in merged_from. "
    'Return ONLY valid JSON: {"gaps": [{"id": "TG-01", "target": "...", "gap_type": "...", '
    '"why_it_matters": "...", "suggested_test": "...", "severity": "critical|high|medium|low", '
    '"merged_from": ["..."]}], "dropped": <how many raw gaps were merged away>}.'
)


def build_dedup_user(raw_json: str) -> str:
    return (
        f"Here are the raw coverage gaps as JSON:\n\n{raw_json}\n\nMerge them into canonical gaps."
    )


SKEPTIC_BASE = (
    "You are an ADVERSARIAL verifier of a claimed test-coverage GAP. Given the SOURCE and its "
    "EXISTING TESTS, decide whether the gap is REAL — the case is genuinely NOT already "
    "covered and is worth a test. Default to real=false unless you can confirm it. "
    'Return ONLY valid JSON: {"real": true|false, "confidence": "high|medium|low", '
    '"severity": "critical|high|medium|low", "reasoning": "one sentence citing the code or tests"}.'
)


def verify_user(context: str, gap_json: str) -> str:
    return (
        f"SOURCE AND EXISTING TESTS:\n\n{context}\n\n"
        f"CLAIMED COVERAGE GAP (JSON):\n{gap_json}\n\n"
        "Is this gap real and worth a test? Return your verdict as JSON now."
    )
