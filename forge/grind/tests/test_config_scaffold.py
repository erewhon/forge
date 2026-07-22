"""Config validation + scaffold round-trip."""

from __future__ import annotations

import pytest

from forge.grind.config import load_config, resolve_model
from forge.grind.models import GrindConfig
from forge.grind.scaffold import DEFAULT_FILENAME, write_skeleton


def test_scaffold_output_validates(tmp_path):
    target = write_skeleton(str(tmp_path))
    assert target.name == DEFAULT_FILENAME
    cfg = load_config(target)  # the skeleton must validate as-is
    assert cfg.goal
    assert [s.name for s in cfg.steps] == ["reset", "load", "migrate"]
    assert cfg.check.score_regex == "RECONCILED=([0-9]+)"


def test_scaffold_refuses_overwrite(tmp_path):
    target = write_skeleton(str(tmp_path / "g.yaml"))
    with pytest.raises(FileExistsError):
        write_skeleton(str(target))
    write_skeleton(str(target), force=True)  # force is allowed


def test_resolved_observe_defaults_to_all_plus_check():
    cfg = GrindConfig(
        goal="g",
        steps=[{"name": "a", "run": "true"}, {"name": "b", "run": "true"}],
        check={"run": "true"},
    )
    assert cfg.resolved_observe() == ["a", "b", "check"]


def test_resolved_observe_rejects_unknown_name():
    cfg = GrindConfig(
        goal="g",
        steps=[{"name": "a", "run": "true"}],
        check={"run": "true"},
        observe=["a", "nope"],
    )
    with pytest.raises(ValueError, match="nope"):
        cfg.resolved_observe()


def test_bad_score_regex_rejected():
    with pytest.raises(ValueError, match="capture group"):
        GrindConfig(
            goal="g",
            steps=[{"name": "a", "run": "true"}],
            check={"run": "true", "score_regex": "no-group-here"},
        )


def test_resolve_model_precedence():
    cfg = GrindConfig(
        goal="g", steps=[{"name": "a", "run": "true"}], check={"run": "true"}, model="from-config"
    )
    assert resolve_model(cfg, "from-cli") == "from-cli"
    assert resolve_model(cfg, None) == "from-config"
    cfg_no_model = cfg.model_copy(update={"model": None})
    assert resolve_model(cfg_no_model, None)  # falls back to the env default, non-empty
