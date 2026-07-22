"""The cheap pre-filter: the AI gate is generous, quadrant classification is a hint, and
word-boundary matching keeps 'agent' from firing on 'management'."""

from __future__ import annotations

from forge.radar.models import Quadrant
from forge.radar.relevance import classify_quadrant, is_relevant, quadrant_scores


def test_relevant_ai_text_passes_the_gate():
    assert is_relevant("Qwen3-Coder: a new open-weights LLM for code")
    assert is_relevant("Building agents with the Model Context Protocol")


def test_irrelevant_text_is_dropped():
    assert not is_relevant("Ask HN: best standing desk for a home office?")
    assert not is_relevant("The economics of high-speed rail in Japan")


def test_word_boundary_prevents_substring_false_positives():
    # "management" contains "agent" as a substring but must not trip the agent term.
    assert not is_relevant("Project management tips for remote teams")
    # But a real agent mention does.
    assert is_relevant("An agent framework for tool use")


def test_multiword_and_dotted_terms_match_as_substrings():
    assert is_relevant("Serving quantized models with llama.cpp")
    assert is_relevant("A guide to the model context protocol")


def test_classify_picks_top_scoring_quadrant():
    assert classify_quadrant("New 30B instruct model, GGUF weights, MoE") is Quadrant.MODELS
    assert (
        classify_quadrant("A multi-agent orchestration framework with tool use") is Quadrant.AGENTS
    )
    assert classify_quadrant("Chain-of-thought prompting and self-consistency evaluation") is (
        Quadrant.TECHNIQUES
    )
    assert classify_quadrant("High-throughput vLLM inference serving on GPU") is Quadrant.INFRA


def test_classify_falls_back_to_hint_when_no_signal():
    assert (
        classify_quadrant("something entirely unrelated", hint=Quadrant.MODELS) is Quadrant.MODELS
    )
    assert classify_quadrant("something entirely unrelated") is None


def test_classify_hint_breaks_ties():
    # Construct text that scores one term in two quadrants; hint decides.
    text = "agent model"  # 'agent' → AGENTS, 'model' → MODELS, tie at 1 each
    scores = quadrant_scores(text)
    assert scores[Quadrant.AGENTS] == 1 and scores[Quadrant.MODELS] == 1
    assert classify_quadrant(text, hint=Quadrant.MODELS) is Quadrant.MODELS
    assert classify_quadrant(text, hint=Quadrant.AGENTS) is Quadrant.AGENTS
