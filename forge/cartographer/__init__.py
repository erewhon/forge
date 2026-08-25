"""``forge map`` — the repo cartographer.

Mirrors configured checkouts from remote VMs (targets are keyed ``(host, path)`` — the same repo
checked out on two VMs is two targets, since checkouts diverge) and emits per-target structural
maps: ``map.md`` for humans, ``index.json`` for the summarizer and renderer stages.

This package is the no-LLM half of the pipeline. The summarizer stage (and with it the
disclosure guards and any scheduled automation) is a separate, later stage — until it lands,
``forge map`` is a hand-run tool.
"""
