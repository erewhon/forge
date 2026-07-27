"""Current-date context for research prompts.

Research harnesses retrieve live sources, so their findings routinely describe events *after* a
model's training cutoff. Without being told today's date, a model falls back on its cutoff as "now"
and misreads correctly-retrieved recent events as fabrication — the adversarial verifier panel was
scoring well-sourced current-events research 2-3/10 with "CRITICAL HALLUCINATION: this is a future
date", and those bogus challenges then steered the next sprint into correcting things that were
right.

The fix is to state the date explicitly, and — for graders — to say plainly that post-cutoff
material is expected and must be judged on *sourcing*, not on whether the model recognises it.
"""

from __future__ import annotations

from datetime import date


def today_line(today: date | None = None) -> str:
    """One line naming today's date, for any prompt that needs to know 'now'."""
    d = today or date.today()
    return f"Today's date is {d:%Y-%m-%d} ({d:%B %d, %Y})."


def researcher_date_context(today: date | None = None) -> str:
    """Date context for a retrieving role (researcher / synthesizer).

    Tells the model that *now* is later than it may assume, so it searches for and reports current
    material instead of hedging to its training cutoff.
    """
    return (
        f"{today_line(today)} Your training data ends earlier than this, so treat your own "
        "knowledge as potentially out of date: rely on what you retrieve this session for anything "
        "time-sensitive, and report retrieved facts even when they postdate what you already know. "
        "Do not describe current events as hypothetical or future."
    )


def verifier_date_context(today: date | None = None) -> str:
    """Date context for a grading role (verifier panel).

    The critical one, and it took two passes to get right. Without any date the panel treats every
    post-cutoff finding as a hallucination, inverting the quality signal. But merely *stating* the
    date was not enough: on a heavily-covered subject the panel would acknowledge "as of the current
    date (July 2026)" and in the same breath call an August 2025 ruling a "future event", then
    assert its own remembered dates ("the actual judgment was issued in November 2023") over the
    cited ones. Strong training priors beat a bare date statement, so the instruction has to forbid
    substituting recollection for citations, not just forbid the word "future".
    """
    return (
        f"{today_line(today)} IMPORTANT: the findings you are grading were produced with live web "
        "search, so they legitimately contain events, dates, appointments, and publications that "
        "postdate your training data. A claim is NOT a hallucination merely because you do not "
        "recognise it or because it is later than your knowledge cutoff — you have no basis to "
        "call a date 'future' or an event 'fabricated' from your own knowledge alone.\n\n"
        "Your own memory is NOT evidence here. Do not correct a cited date, ruling, holding, or "
        "outcome using your recollection: on subjects with heavy pre-cutoff coverage your priors "
        "describe an earlier state of the world and the findings may describe a later one. If you "
        "believe a cited fact is wrong, you must point to the specific source in the findings that "
        "contradicts it, or say the citation does not support the claim — never assert a competing "
        "fact from memory, and never label something fabricated because it disagrees with what you "
        "remember.\n\n"
        "Judge each claim on whether the findings cite a source that actually supports it. "
        "Unsourced, vague, and source-mismatched claims are still fair game and you should attack "
        "them hard; an answer that does not address the question it was asked is fair game; "
        "recency is not."
    )
