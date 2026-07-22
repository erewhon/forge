"""The cheap relevance pre-filter and quadrant classifier — volume control for the scanner firehose.

This layer is deliberately dumb. Its only job is to keep obvious noise out of the candidate feed
(a Hacker News front page is mostly *not* about AI) and attach a provisional quadrant hint. The
**real** relevance and placement judgment happens in the weekly synthesis, which reads the feed with
a strong model; nothing here should try to be clever, and false positives are fine — a borderline
candidate reaching synthesis is cheap, a good one silently dropped here is not. So the gate is
generous and the classifier only ever produces a *hint*.

Matching is whole-word for single tokens (so "agent" does not fire on "management") and substring
for multi-word phrases. The lexicon is intentionally stack-personal — weighted toward what actually
changes how *this* stack builds (local models, agent frameworks, inference/routing tooling), not a
generic AI-news filter.
"""

from __future__ import annotations

import re

from forge.radar.models import Quadrant

#: Broad AI/LLM/agent/tooling gate. A candidate whose title+summary contains none of these is
#: dropped before it reaches the feed. Generous by design — synthesis does the real filtering.
RELEVANCE_TERMS: frozenset[str] = frozenset(
    {
        "llm",
        "language model",
        "gpt",
        "claude",
        "anthropic",
        "openai",
        "gemini",
        "mistral",
        "qwen",
        "llama",
        "deepseek",
        "moonshot",
        "kimi",
        "minimax",
        "gemma",
        "phi",
        "mixtral",
        "transformer",
        "diffusion",
        "embedding",
        "rag",
        "retrieval-augmented",
        "inference",
        "quantization",
        "quantized",
        "gguf",
        "fine-tune",
        "finetune",
        "lora",
        "agent",
        "agentic",
        "mcp",
        "model context protocol",
        "prompt",
        "prompting",
        "eval",
        "benchmark",
        "reasoning",
        "tool call",
        "tool-calling",
        "function calling",
        "context window",
        "vllm",
        "sglang",
        "ollama",
        "llama.cpp",
        "tokenizer",
        "multimodal",
        "vision-language",
        "openai-compatible",
        "copilot",
        "coder",
        "code model",
        "chatbot",
        "fine-tuning",
        "distillation",
        "rlhf",
        # Common HuggingFace pipeline tags / model descriptors — LLM-specific, so they widen the
        # gate for genuine language models without loosening it for the un-scoped HN firehose.
        "text-generation",
        "text2text",
        "conversational",
        "instruct",
        "chat",
        "code generation",
        "text-to-code",
    }
)

#: Per-quadrant signal terms. A candidate is scored by how many terms of each quadrant it hits; the
#: top-scoring quadrant becomes the hint (ties and empties fall back to the adapter's own hint).
QUADRANT_TERMS: dict[Quadrant, frozenset[str]] = {
    Quadrant.MODELS: frozenset(
        {
            "model",
            "llm",
            "gpt",
            "claude",
            "gemini",
            "qwen",
            "llama",
            "mistral",
            "deepseek",
            "kimi",
            "minimax",
            "gemma",
            "phi",
            "mixtral",
            "checkpoint",
            "weights",
            "quant",
            "quantized",
            "gguf",
            "lora",
            "instruct",
            "moe",
            "parameters",
            "7b",
            "13b",
            "30b",
            "70b",
            "multimodal",
            "vision-language",
            "base model",
            "distillation",
        }
    ),
    Quadrant.AGENTS: frozenset(
        {
            "agent",
            "agentic",
            "framework",
            "orchestration",
            "langchain",
            "langgraph",
            "autogen",
            "crewai",
            "swarm",
            "multi-agent",
            "tool use",
            "tool-calling",
            "mcp",
            "model context protocol",
            "workflow",
            "sdk",
            "harness",
            "copilot",
            "assistant",
            "function calling",
        }
    ),
    Quadrant.TECHNIQUES: frozenset(
        {
            "prompt",
            "prompting",
            "chain-of-thought",
            "rag",
            "retrieval",
            "retrieval-augmented",
            "fine-tuning",
            "distillation",
            "rlhf",
            "dpo",
            "evaluation",
            "benchmark",
            "reasoning",
            "in-context",
            "few-shot",
            "technique",
            "method",
            "attention",
            "scaling law",
            "self-consistency",
            "verifier",
        }
    ),
    Quadrant.INFRA: frozenset(
        {
            "inference",
            "serving",
            "vllm",
            "sglang",
            "ollama",
            "llama.cpp",
            "gpu",
            "cuda",
            "router",
            "gateway",
            "deployment",
            "container",
            "throughput",
            "latency",
            "kv cache",
            "vector database",
            "observability",
            "tooling",
            "cli",
            "openai-compatible",
            "runtime",
            "quantization",
        }
    ),
}


def _contains(text: str, term: str) -> bool:
    """Whole-word match for single tokens; substring match for multi-word phrases. Keeps "agent"
    from firing on "management" while still matching "llama.cpp" and "model context protocol"."""
    if " " in term or "." in term:
        return term in text
    return re.search(rf"\b{re.escape(term)}\b", text) is not None


def is_relevant(text: str) -> bool:
    """Whether *text* (a title + summary) trips the broad AI gate. Generous on purpose."""
    lowered = text.lower()
    return any(_contains(lowered, term) for term in RELEVANCE_TERMS)


def quadrant_scores(text: str) -> dict[Quadrant, int]:
    """How many terms of each quadrant *text* hits."""
    lowered = text.lower()
    return {
        quad: sum(_contains(lowered, term) for term in terms)
        for quad, terms in QUADRANT_TERMS.items()
    }


def classify_quadrant(text: str, hint: Quadrant | None = None) -> Quadrant | None:
    """A provisional quadrant for *text*: the top-scoring quadrant, with the adapter's *hint*
    breaking ties and standing in when nothing scores. Returns ``None`` only when there is no signal
    and no hint — the caller may then drop the candidate or file it unclassified. This is a hint for
    synthesis, never a final placement."""
    scores = quadrant_scores(text)
    best = max(scores.values())
    if best == 0:
        return hint
    winners = [q for q, s in scores.items() if s == best]
    if len(winners) == 1:
        return winners[0]
    # Tie: prefer the adapter's hint if it is among the winners, else the first by quadrant order.
    if hint in winners:
        return hint
    return winners[0]
