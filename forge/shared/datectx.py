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

    The critical one. Without it the panel treats every post-cutoff finding as a hallucination,
    which inverts the quality signal: the better the retrieval, the worse the score.
    """
    return (
        f"{today_line(today)} IMPORTANT: the findings you are grading were produced with live web "
        "search, so they legitimately contain events, dates, appointments, and publications that "
        "postdate your training data. A claim is NOT a hallucination merely because you do not "
        "recognise it or because it is later than your knowledge cutoff — you have no basis to "
        "call a date 'future' or an event 'fabricated' from your own knowledge alone. Judge each "
        "claim on whether the findings cite a source that actually supports it. Unsourced, vague, "
        "source-mismatched claims are still fair game and you should attack them; recency is not."
    )
