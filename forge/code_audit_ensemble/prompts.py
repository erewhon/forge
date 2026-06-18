"""Prompts for the code-audit ensemble: blind finder angles, the consolidator, and skeptic lenses.

The code context is fed in the *user* message (finders read provided code — the harness executors
have no file tools); each finder's *system* prompt aims it at one angle, so the union of findings
covers orthogonal failure modes. Skeptic lenses give the verification panel diverse ways to refute.
"""

from __future__ import annotations

# Each finder hunts ONE class of problem hardest (still reporting anything it sees). The union
# across angles covers more than any single reviewer would, before dedup collapses the overlap.
FINDER_ANGLES: tuple[tuple[str, str], ...] = (
    (
        "correctness",
        "logic errors, wrong conditions, off-by-one, mishandled return values, broken invariants",
    ),
    (
        "error-handling",
        "swallowed/over-broad exceptions, unchecked failures, errors that leave inconsistent "
        "state, missing rollback",
    ),
    (
        "edge-cases",
        "empty/None/zero inputs, boundary values, unicode, concurrency/ordering assumptions, "
        "unhandled states",
    ),
    (
        "resource-safety",
        "leaked files/sockets/locks, unbounded growth, missing timeouts, blocking calls, cleanup "
        "that doesn't run on failure",
    ),
    (
        "security",
        "injection, unsafe deserialization, path traversal, secrets in code, missing authz, unsafe "
        "use of untrusted input",
    ),
)

# Adversarial verification lenses — each skeptic tries to REFUTE the finding from a different angle.
SKEPTIC_LENSES: tuple[tuple[str, str], ...] = (
    (
        "reachability",
        "Can a real caller actually reach this? If a guard, type, or caller contract already "
        "prevents it, it is not real.",
    ),
    (
        "code-accuracy",
        "Does the code truly behave as the finding claims? Watch for misreads and wrong line "
        "citations; if the described behavior isn't what the code does, it is not real.",
    ),
)

_FINDING_SHAPE = (
    'Return ONLY valid JSON: {"findings": [{"title": "...", "file": "...", "line": "...", '
    '"severity": "critical|high|medium|low", "scenario": "the concrete problem and how it bites", '
    '"suggestion": "the fix"}]}. Return an empty findings list if you find nothing real.'
)


def finder_system(focus: str, angle_directive: str) -> str:
    return (
        "You are a code auditor reviewing the code provided by the user. "
        f"AUDIT FOCUS: {focus}. "
        f"YOUR ANGLE — hunt hardest for: {angle_directive}. "
        "Report only CONCRETE, REACHABLE problems you can point to in the provided code; never "
        "infer from names alone. Be specific about the file and line. Do not invent issues to "
        "fill a quota. " + _FINDING_SHAPE
    )


DEDUP_SYSTEM = (
    "You consolidate code-audit findings from several independent reviewers of the SAME code. "
    "Merge findings that describe the same root cause / same code location into ONE canonical "
    "finding; keep genuinely distinct issues separate. For each canonical finding assign a stable "
    "id (CA-01, CA-02, ...), keep the clearest title/scenario/suggestion, take the HIGHEST "
    "severity among the merged duplicates, and list the merged source titles in merged_from. "
    'Return ONLY valid JSON: {"findings": [{"id": "CA-01", "title": "...", "file": "...", '
    '"line": "...", "severity": "critical|high|medium|low", "scenario": "...", '
    '"suggestion": "...", "merged_from": ["..."]}], '
    '"dropped": <how many raw findings were merged away>}.'
)


def build_dedup_user(raw_json: str) -> str:
    return (
        f"Here are the raw findings as JSON:\n\n{raw_json}\n\nMerge them into canonical findings."
    )


SKEPTIC_BASE = (
    "You are an ADVERSARIAL verifier of a claimed code-audit finding. Your job is to REFUTE it. "
    "Read the provided code and decide whether the finding describes a CONCRETE, REACHABLE "
    "problem. Default to real=false unless you can confirm it. If it is real but mis-rated, set "
    "real=true and correct the severity. "
    'Return ONLY valid JSON: {"real": true|false, "confidence": "high|medium|low", '
    '"severity": "critical|high|medium|low", "reasoning": "one sentence citing the code"}.'
)


def verify_user(code_context: str, finding_json: str) -> str:
    return (
        f"CODE UNDER AUDIT:\n\n{code_context}\n\n"
        f"CLAIMED FINDING (JSON):\n{finding_json}\n\n"
        "Is this a real, reachable problem? Return your verdict as JSON now."
    )
