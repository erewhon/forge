from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agents.evals.fixtures import EvalFixtureError, load_goldsets, read_input

# -- helpers --


def _write_case_yaml(case_dir: Path, data: dict) -> None:
    (case_dir / "case.yaml").write_text(yaml.dump(data), encoding="utf-8")


def _make_fixture_root(tmp_path: Path, cases: list[dict]) -> Path:
    """Create a goldsets directory with *cases* sub-directories.

    Each entry in *cases* is a dict with keys:
        name (str): sub-directory name (also used as step)
        yaml (dict): case.yaml contents
        files (list[str]): extra files to create in the case dir
    """
    goldsets = tmp_path / "goldsets"
    goldsets.mkdir()
    for c in cases:
        case_dir = goldsets / c["name"]
        case_dir.mkdir()
        _write_case_yaml(case_dir, c["yaml"])
        for f in c.get("files", []):
            (case_dir / f).write_text("content", encoding="utf-8")
    return goldsets


# -- happy path --


def test_load_goldsets_basic(tmp_path: Path):
    root = _make_fixture_root(
        tmp_path,
        [
            {
                "name": "replan",
                "yaml": {
                    "schema_version": 1,
                    "step": "replan",
                    "holdout": False,
                    "inputs": {"diff": "a.patch"},
                    "expected": {"action": "replan"},
                    "notes": "basic replan case",
                },
                "files": ["a.patch"],
            }
        ],
    )
    cases = load_goldsets(root)
    assert len(cases) == 1
    c = cases[0]
    assert c.step == "replan"
    assert c.case_id == "replan"
    assert c.holdout is False
    assert c.inputs == {"diff": "a.patch"}
    assert c.expected == {"action": "replan"}
    assert c.notes == "basic replan case"
    assert c.schema_version == 1


def test_load_goldsets_deterministic_order(tmp_path: Path):
    root = _make_fixture_root(
        tmp_path,
        [
            {
                "name": "review-findings",
                "yaml": {
                    "schema_version": 1,
                    "step": "review-findings",
                    "inputs": {"diff": "d.patch"},
                },
                "files": ["d.patch"],
            },
            {
                "name": "decompose",
                "yaml": {
                    "schema_version": 1,
                    "step": "decompose",
                    "inputs": {"req": "r.txt"},
                },
                "files": ["r.txt"],
            },
        ],
    )
    cases = load_goldsets(root)
    # Should be sorted by (step, case_id)
    assert cases[0].step == "decompose"
    assert cases[1].step == "review-findings"


def test_load_goldsets_holdout(tmp_path: Path):
    root = _make_fixture_root(
        tmp_path,
        [
            {
                "name": "replan",
                "yaml": {
                    "schema_version": 1,
                    "step": "replan",
                    "holdout": True,
                },
                "files": [],
            }
        ],
    )
    cases = load_goldsets(root)
    assert cases[0].holdout is True


def test_load_goldsets_filter_by_step(tmp_path: Path):
    root = _make_fixture_root(
        tmp_path,
        [
            {
                "name": "replan",
                "yaml": {"schema_version": 1, "step": "replan"},
                "files": [],
            },
            {
                "name": "decompose",
                "yaml": {"schema_version": 1, "step": "decompose"},
                "files": [],
            },
        ],
    )
    cases = load_goldsets(root, step="replan")
    assert len(cases) == 1
    assert cases[0].step == "replan"


def test_load_goldsets_multiple_dirs(tmp_path: Path):
    root = _make_fixture_root(
        tmp_path,
        [
            {
                "name": "replan",
                "yaml": {
                    "schema_version": 1,
                    "step": "replan",
                    "inputs": {"diff": "a.patch"},
                },
                "files": ["a.patch"],
            },
            {
                "name": "decompose",
                "yaml": {
                    "schema_version": 1,
                    "step": "decompose",
                    "inputs": {"req": "r.txt"},
                },
                "files": ["r.txt"],
            },
        ],
    )
    cases = load_goldsets(root)
    assert len(cases) == 2


def test_load_goldsets_empty_goldsets_dir(tmp_path: Path):
    root = _make_fixture_root(tmp_path, [])
    cases = load_goldsets(root)
    assert cases == []


# -- error cases --


def test_invalid_step_name_raises_validation(tmp_path: Path):
    """A case whose step is not a valid StepName fails at GoldCase construction."""
    root = _make_fixture_root(
        tmp_path,
        [
            {
                "name": "replan",
                "yaml": {
                    "schema_version": 1,
                    "step": "replan",
                },
                "files": [],
            }
        ],
    )
    # Normal case loads fine
    cases = load_goldsets(root)
    assert len(cases) == 1


def test_missing_case_yaml(tmp_path: Path):
    goldsets = tmp_path / "goldsets"
    goldsets.mkdir()
    case_dir = goldsets / "replan"
    case_dir.mkdir()
    # No case.yaml

    with pytest.raises(EvalFixtureError, match="missing case.yaml.*replan"):
        load_goldsets(goldsets)


def test_step_dir_mismatch(tmp_path: Path):
    root = _make_fixture_root(
        tmp_path,
        [
            {
                "name": "replan",
                "yaml": {
                    "schema_version": 1,
                    "step": "decompose",  # mismatch!
                },
                "files": [],
            }
        ],
    )

    with pytest.raises(EvalFixtureError, match="does not match directory name"):
        load_goldsets(root)


def test_wrong_schema_version(tmp_path: Path):
    root = _make_fixture_root(
        tmp_path,
        [
            {
                "name": "replan",
                "yaml": {
                    "schema_version": 2,
                    "step": "replan",
                },
                "files": [],
            }
        ],
    )

    with pytest.raises(EvalFixtureError, match="unsupported schema_version"):
        load_goldsets(root)


def test_missing_input_file(tmp_path: Path):
    root = _make_fixture_root(
        tmp_path,
        [
            {
                "name": "replan",
                "yaml": {
                    "schema_version": 1,
                    "step": "replan",
                    "inputs": {"diff": "missing.patch"},
                },
                "files": [],  # file not created
            }
        ],
    )

    with pytest.raises(EvalFixtureError, match="input file.*not found"):
        load_goldsets(root)


def test_goldsets_root_not_directory(tmp_path: Path):
    non_existent = tmp_path / "nonexistent"
    with pytest.raises(EvalFixtureError, match="not a directory"):
        load_goldsets(non_existent)


# -- read_input --


def test_read_input_success(tmp_path: Path):
    root = _make_fixture_root(
        tmp_path,
        [
            {
                "name": "replan",
                "yaml": {
                    "schema_version": 1,
                    "step": "replan",
                    "inputs": {"diff": "a.patch"},
                },
                "files": ["a.patch"],
            }
        ],
    )
    cases = load_goldsets(root)
    case = cases[0]
    content = read_input(case, "diff")
    assert content == "content"


def test_read_input_missing_name(tmp_path: Path):
    root = _make_fixture_root(
        tmp_path,
        [
            {
                "name": "replan",
                "yaml": {
                    "schema_version": 1,
                    "step": "replan",
                    "inputs": {"diff": "a.patch"},
                },
                "files": ["a.patch"],
            }
        ],
    )
    cases = load_goldsets(root)
    case = cases[0]

    with pytest.raises(EvalFixtureError, match="not found in case inputs"):
        read_input(case, "nonexistent")


def test_read_input_case_dir_attribute(tmp_path: Path):
    root = _make_fixture_root(
        tmp_path,
        [
            {
                "name": "replan",
                "yaml": {
                    "schema_version": 1,
                    "step": "replan",
                    "inputs": {"diff": "a.patch"},
                },
                "files": ["a.patch"],
            }
        ],
    )
    cases = load_goldsets(root)
    case = cases[0]
    assert case.case_dir == root / "replan"
