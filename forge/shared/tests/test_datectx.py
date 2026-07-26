"""Date context injected into research prompts.

Regression cover for the failure that motivated it: the verifier panel, given no date, judged
correctly-retrieved post-cutoff events against its own training cutoff and scored well-sourced
current-events research 2-3/10 with "CRITICAL HALLUCINATION: this is a future date" — inverting the
quality signal, since better retrieval produced worse scores.
"""

from __future__ import annotations

from datetime import date

from forge.book_researcher import planner as book_planner
from forge.book_researcher import researcher as book_researcher
from forge.book_researcher import verifier as book_verifier
from forge.general_researcher import planner as gen_planner
from forge.general_researcher import researcher as gen_researcher
from forge.general_researcher import synthesizer as gen_synthesizer
from forge.general_researcher import verifier as gen_verifier
from forge.shared.datectx import researcher_date_context, today_line, verifier_date_context

FIXED = date(2026, 7, 25)


def test_today_line_states_the_date():
    line = today_line(FIXED)
    assert "2026-07-25" in line
    assert "July 25, 2026" in line


def test_verifier_context_forbids_calling_recency_hallucination():
    ctx = verifier_date_context(FIXED).lower()
    assert "2026-07-25" in ctx
    # the load-bearing instruction: recency is not evidence of fabrication
    assert "not a hallucination" in ctx
    assert "live web search" in ctx
    # ...but it must stay adversarial about actual sourcing failures
    assert "unsourced" in ctx


def test_researcher_context_says_own_knowledge_may_be_stale():
    ctx = researcher_date_context(FIXED).lower()
    assert "2026-07-25" in ctx
    assert "out of date" in ctx
    assert "postdate" in ctx


def test_context_defaults_to_real_today():
    assert date.today().strftime("%Y-%m-%d") in verifier_date_context()
    assert date.today().strftime("%Y-%m-%d") in researcher_date_context()


def test_every_research_role_imports_the_context():
    """Each LLM-facing role must carry date context — a role that misses it reintroduces the bug."""
    for mod, fn in (
        (gen_verifier, "verifier_date_context"),
        (book_verifier, "verifier_date_context"),
        (gen_researcher, "researcher_date_context"),
        (book_researcher, "researcher_date_context"),
        (gen_synthesizer, "researcher_date_context"),
        (gen_planner, "researcher_date_context"),
        (book_planner, "researcher_date_context"),
    ):
        assert hasattr(mod, fn), f"{mod.__name__} does not import {fn}"
