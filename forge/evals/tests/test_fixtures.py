from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from forge.evals.fixtures import EvalFixtureError, load_goldsets, read_input

# -- helpers --


def _write_case_yaml(case_dir: Path, data: dict) -> None:
    (case_dir / "case.yaml").write_text(yaml.dump(data), encoding="utf-8")


def _make_fixture_root(tmp_path: Path, cases: list[dict]) -> Path:
    """Create a ``goldsets/<step>/<case>/`` tree from *cases*.

    Each entry in *cases* is a dict with keys:
        step (str): step directory name
        case (str): case directory name (the case_id)
        yaml (dict): case.yaml contents
        files (list[str]): extra files to create in the case dir
    """
    goldsets = tmp_path / "goldsets"
    goldsets.mkdir()
    for c in cases:
        case_dir = goldsets / c["step"] / c["case"]
        case_dir.mkdir(parents=True)
        _write_case_yaml(case_dir, c["yaml"])
        for f in c.get("files", []):
            (case_dir / f).write_text("content", encoding="utf-8")
    return goldsets


def _case(step: str, case: str, *, files: list[str] | None = None, **yaml_extra) -> dict:
    data = {"schema_version": 1, "step": step, **yaml_extra}
    if files:
        data.setdefault("inputs", {f.split(".")[0]: f for f in files})
    return {"step": step, "case": case, "yaml": data, "files": files or []}


# -- happy path --


def test_load_goldsets_basic(tmp_path: Path):
    root = _make_fixture_root(
        tmp_path,
        [
            {
                "step": "replan",
                "case": "fixup-confirmed",
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
    assert c.case_id == "fixup-confirmed"
    assert c.case_dir == root / "replan" / "fixup-confirmed"
    assert c.holdout is False
    assert c.inputs == {"diff": "a.patch"}
    assert c.expected == {"action": "replan"}
    assert c.notes == "basic replan case"
    assert c.schema_version == 1


def test_load_goldsets_multiple_cases_per_step(tmp_path: Path):
    """A step directory holds any number of case directories — the whole point."""
    root = _make_fixture_root(
        tmp_path,
        [
            _case("replan", "case-b"),
            _case("replan", "case-a"),
            _case("replan", "case-c", holdout=True),
        ],
    )
    cases = load_goldsets(root, step="replan")
    assert [c.case_id for c in cases] == ["case-a", "case-b", "case-c"]
    assert all(c.step == "replan" for c in cases)
    assert [c.holdout for c in cases] == [False, False, True]


def test_load_goldsets_deterministic_order(tmp_path: Path):
    root = _make_fixture_root(
        tmp_path,
        [
            _case("review-findings", "z-case", files=["d.patch"]),
            _case("decompose", "a-case", files=["r.txt"]),
        ],
    )
    cases = load_goldsets(root)
    # Should be sorted by (step, case_id)
    assert cases[0].step == "decompose"
    assert cases[1].step == "review-findings"


def test_load_goldsets_holdout(tmp_path: Path):
    root = _make_fixture_root(tmp_path, [_case("replan", "held", holdout=True)])
    cases = load_goldsets(root)
    assert cases[0].holdout is True


def test_load_goldsets_filter_by_step(tmp_path: Path):
    root = _make_fixture_root(
        tmp_path,
        [
            _case("replan", "one"),
            _case("decompose", "two"),
        ],
    )
    cases = load_goldsets(root, step="replan")
    assert len(cases) == 1
    assert cases[0].step == "replan"
    assert cases[0].case_id == "one"


def test_load_goldsets_multiple_steps(tmp_path: Path):
    root = _make_fixture_root(
        tmp_path,
        [
            _case("replan", "one", files=["a.patch"]),
            _case("decompose", "two", files=["r.txt"]),
        ],
    )
    cases = load_goldsets(root)
    assert len(cases) == 2


def test_load_goldsets_empty_goldsets_dir(tmp_path: Path):
    root = _make_fixture_root(tmp_path, [])
    cases = load_goldsets(root)
    assert cases == []


def test_load_goldsets_empty_step_dir(tmp_path: Path):
    goldsets = tmp_path / "goldsets"
    (goldsets / "replan").mkdir(parents=True)
    assert load_goldsets(goldsets) == []


# -- error cases --


def test_missing_case_yaml(tmp_path: Path):
    goldsets = tmp_path / "goldsets"
    case_dir = goldsets / "replan" / "broken"
    case_dir.mkdir(parents=True)
    # No case.yaml

    with pytest.raises(EvalFixtureError, match="missing case.yaml.*broken"):
        load_goldsets(goldsets)


def test_step_dir_mismatch(tmp_path: Path):
    root = _make_fixture_root(
        tmp_path,
        [
            {
                "step": "replan",
                "case": "mismatched",
                "yaml": {"schema_version": 1, "step": "decompose"},  # mismatch!
                "files": [],
            }
        ],
    )

    with pytest.raises(EvalFixtureError, match="does not match step directory"):
        load_goldsets(root)


def test_wrong_schema_version(tmp_path: Path):
    root = _make_fixture_root(
        tmp_path,
        [
            {
                "step": "replan",
                "case": "future",
                "yaml": {"schema_version": 2, "step": "replan"},
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
                "step": "replan",
                "case": "no-input",
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
    root = _make_fixture_root(tmp_path, [_case("replan", "one", files=["a.patch"])])
    case = load_goldsets(root)[0]
    assert read_input(case, "a") == "content"


def test_read_input_missing_name(tmp_path: Path):
    root = _make_fixture_root(tmp_path, [_case("replan", "one", files=["a.patch"])])
    case = load_goldsets(root)[0]

    with pytest.raises(EvalFixtureError, match="not found in case inputs"):
        read_input(case, "nonexistent")


def test_read_input_case_dir_attribute(tmp_path: Path):
    root = _make_fixture_root(tmp_path, [_case("replan", "one", files=["a.patch"])])
    case = load_goldsets(root)[0]
    assert case.case_dir == root / "replan" / "one"
