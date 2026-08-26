"""Render assembly from the cache, and the full `run` pipeline with failure isolation."""

from __future__ import annotations

import configparser
from pathlib import Path

import pytest

from forge.cartographer.models import MapConfig, TargetConfig
from forge.cartographer.render import render_target, run_all
from forge.cartographer.summarize import summarize_target
from forge.cartographer.tests.test_summarize import (
    _SHAS,
    CountingModel,
    _config,
    _settings,
    _write_target,
)

_UNITS = Path(__file__).resolve().parents[1] / "systemd"


def test_render_assembles_docs_from_cache(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_target(config, config.targets[0], _SHAS)
    summarize_target(config, _settings(), config.targets[0], CountingModel(), CountingModel())

    stats = render_target(config, config.targets[0])
    out = config.out_dir(config.targets[0])
    assert stats.modules_written == 2 and stats.architecture_written
    assert stats.files_with_summary == 3 and not stats.files_missing_summary
    assert "a summary." in (out / "modules" / "app.md").read_text()
    assert (out / "architecture.md").read_text().startswith("# Architecture — t1")
    assert (out / "map.md").is_file()


def test_render_marks_missing_summaries_not_silently(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_target(config, config.targets[0], _SHAS)
    failing = CountingModel()
    failing.reply = ""  # nothing gets cached
    summarize_target(config, _settings(), config.targets[0], failing, CountingModel())

    stats = render_target(config, config.targets[0])
    assert sorted(stats.files_missing_summary) == sorted(_SHAS)
    assert not stats.architecture_written
    text = (config.out_dir(config.targets[0]) / "modules" / "app.md").read_text()
    assert "(no summary yet)" in text


def test_render_without_summarize_is_a_clear_error(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_target(config, config.targets[0], _SHAS)
    (config.out_dir(config.targets[0]) / "summaries.json").unlink(missing_ok=True)
    with pytest.raises(FileNotFoundError, match="forge map summarize"):
        render_target(config, config.targets[0])


def test_run_end_to_end_and_failure_isolation(tmp_path: Path, mini_repo: Path) -> None:
    pytest.importorskip("tree_sitter_language_pack", reason="map extra not installed")
    config = MapConfig(
        targets=[
            TargetConfig(name="good", path=str(mini_repo)),  # local → real rsync
            TargetConfig(name="down", host="host.invalid", path="/nowhere"),
        ],
        output_root=tmp_path / "maps",
    )
    exit_code, reports = run_all(
        config, _settings(), config.targets, CountingModel(), CountingModel()
    )
    by_name = {r.name: r for r in reports}
    assert by_name["good"].ok and by_name["good"].parsed > 0
    assert not by_name["down"].ok and by_name["down"].error
    assert exit_code == 0, "one live target means the run as a whole succeeded"

    out = config.out_dir(config.targets[0])
    for artifact in ("map.md", "index.json", "summaries.json", "architecture.md"):
        assert (out / artifact).is_file()
    report = (config.output_root / "report.md").read_text()
    assert "## good: OK" in report and "## down: FAILED" in report
    assert f"summaries: {by_name['good'].summarize.summarized} new" in report


def test_run_all_targets_failing_is_nonzero(tmp_path: Path) -> None:
    config = MapConfig(
        targets=[TargetConfig(name="down", host="host.invalid", path="/nowhere")],
        output_root=tmp_path / "maps",
    )
    exit_code, reports = run_all(
        config, _settings(), config.targets, CountingModel(), CountingModel()
    )
    assert exit_code == 1 and not reports[0].ok


@pytest.mark.parametrize("unit", ["forge-map.service", "forge-map.timer"])
def test_systemd_units_parse(unit: str) -> None:
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    parser.read(_UNITS / unit)
    if unit.endswith(".service"):
        assert parser["Service"]["Type"] == "oneshot"
        assert parser["Service"]["ExecStart"].endswith("forge map run")
    else:
        assert parser["Timer"]["OnCalendar"] == "*-*-* 03:30:00"
        assert parser["Timer"]["Persistent"] == "true"
    assert "Install" in parser
