"""Adapter parsers — each pure ``parse`` against a small captured-shape fixture. No network."""

from __future__ import annotations

from forge.radar.models import Quadrant
from forge.radar.sources import default_adapters
from forge.radar.sources.arxiv import ArxivAdapter
from forge.radar.sources.github import GitHubAdapter
from forge.radar.sources.hackernews import HackerNewsAdapter
from forge.radar.sources.huggingface import HuggingFaceAdapter


def test_default_adapters_are_the_four_clean_sources():
    names = {a.name for a in default_adapters()}
    assert names == {"huggingface", "github", "arxiv", "hackernews"}


def test_huggingface_parse():
    payload = [
        {
            "id": "Qwen/Qwen3-Coder-30B",
            "likes": 512,
            "pipeline_tag": "text-generation",
            "tags": ["code", "moe"],
            "createdAt": "2026-07-01T00:00:00Z",
        },
        {"modelId": "org/no-id-field"},  # id under modelId; no likes
        {"likes": 5},  # no id at all → skipped
    ]
    items = HuggingFaceAdapter().parse(payload)
    assert [i.external_id for i in items] == ["Qwen/Qwen3-Coder-30B", "org/no-id-field"]
    first = items[0]
    assert first.source == "huggingface"
    assert first.url == "https://huggingface.co/Qwen/Qwen3-Coder-30B"
    assert first.quadrant_hint is Quadrant.MODELS
    assert first.score == 512.0
    assert "text-generation" in first.summary and "code" in first.summary


def test_github_parse_leaves_quadrant_to_the_classifier():
    payload = {
        "items": [
            {
                "full_name": "langchain-ai/langgraph",
                "html_url": "https://github.com/langchain-ai/langgraph",
                "description": "agent orchestration",
                "stargazers_count": 9000,
                "pushed_at": "2026-07-20T00:00:00Z",
            },
            {"description": "no full_name → skipped"},
        ]
    }
    items = GitHubAdapter().parse(payload)
    assert len(items) == 1
    it = items[0]
    assert it.external_id == "langchain-ai/langgraph"
    assert it.quadrant_hint is None  # classifier decides from the description
    assert it.score == 9000.0


def test_arxiv_parse_atom_xml():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2507.01234v2</id>
        <title>A New
          Reasoning Technique</title>
        <summary>  We propose   a method
          for better reasoning.  </summary>
        <published>2026-07-15T00:00:00Z</published>
      </entry>
    </feed>"""
    items = ArxivAdapter().parse(xml)
    assert len(items) == 1
    it = items[0]
    assert it.external_id == "2507.01234"  # version stripped
    assert it.title == "A New Reasoning Technique"  # whitespace flattened
    assert it.summary == "We propose a method for better reasoning."
    assert it.quadrant_hint is Quadrant.TECHNIQUES
    assert it.score is None


def test_hackernews_parse_falls_back_to_item_url():
    payload = {
        "hits": [
            {
                "objectID": "111",
                "title": "Show HN: an LLM agent",
                "url": "https://example.com/x",
                "points": 240,
                "created_at": "2026-07-20T00:00:00Z",
            },
            {
                "objectID": "222",
                "title": "Text post, no url",
                "points": 10,
            },  # no url → HN item page
            {"points": 5},  # no objectID/title → skipped
        ]
    }
    items = HackerNewsAdapter().parse(payload)
    assert [i.external_id for i in items] == ["111", "222"]
    assert items[0].url == "https://example.com/x"
    assert items[1].url == "https://news.ycombinator.com/item?id=222"
    assert items[0].quadrant_hint is None
    assert items[0].score == 240.0
