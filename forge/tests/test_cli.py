"""Tests for the `meta` CLI dispatcher — verbs route to agent main(argv), no agents run."""

from __future__ import annotations

from typer.testing import CliRunner

import agents.registry as registry
from agents.cli import app

runner = CliRunner()


def test_help_lists_all_registered_verbs():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in registry.REGISTRY:
        assert cmd.name in result.output


def test_verb_forwards_extra_argv_to_agent_main(monkeypatch):
    captured: dict = {}

    def fake_main(argv):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr("agents.general_researcher.main.main", fake_main)
    result = runner.invoke(app, ["research", "why X?", "--max-sprints", "3", "--dry-run"])
    assert result.exit_code == 0
    # the verb token is consumed; everything after it is handed to the agent's own parser
    assert captured["argv"] == ["why X?", "--max-sprints", "3", "--dry-run"]


def test_agent_exit_code_propagates(monkeypatch):
    monkeypatch.setattr("agents.pr_review_ensemble.main.main", lambda argv: 2)
    result = runner.invoke(app, ["review", "--pass", "review"])
    assert result.exit_code == 2


def test_none_return_maps_to_zero(monkeypatch):
    monkeypatch.setattr("agents.parallel_edit.main.main", lambda argv: None)
    result = runner.invoke(app, ["edit", "--prompt", "x"])
    assert result.exit_code == 0


def test_unknown_verb_errors():
    result = runner.invoke(app, ["does-not-exist"])
    assert result.exit_code != 0


def test_help_does_not_import_any_agent(monkeypatch):
    # Lazy loading: rendering --help must not import a single agent module.
    real_import = registry.importlib.import_module
    seen: list[str] = []

    def spy(name, *args, **kwargs):
        seen.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(registry.importlib, "import_module", spy)
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert not any(cmd.module in seen for cmd in registry.REGISTRY)


def test_every_registered_agent_exposes_a_callable_main():
    # Guards against a registry entry pointing at an agent that lacks a main(argv) entry point.
    for cmd in registry.REGISTRY:
        assert callable(cmd.load_main()), f"{cmd.name} -> {cmd.module} has no callable main"


def test_normalized_no_main_agent_routes(monkeypatch):
    # astro_scout had no main() (it parsed sys.argv[1] directly); the new main(argv) must route.
    captured: dict = {}

    def fake_main(argv):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr("agents.astro_scout.main.main", fake_main)
    result = runner.invoke(app, ["astro", "2026-06-18"])
    assert result.exit_code == 0
    assert captured["argv"] == ["2026-06-18"]
