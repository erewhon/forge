from __future__ import annotations

from pathlib import Path

import yaml

from forge.evals.models import GoldCase

SUPPORTED_SCHEMA_VERSION = 1


class EvalFixtureError(RuntimeError):
    """Raised when a gold case fixture is invalid."""

    def __init__(self, message: str, case_dir: Path | None = None) -> None:
        self.case_dir = case_dir
        if case_dir:
            message = f"{message} (case: {case_dir})"
        super().__init__(message)


def load_goldsets(root: Path, step: str | None = None) -> list[GoldCase]:
    """Walk ``root/<step>/<case-id>/`` and yield validated :class:`GoldCase` objects.

    Each step directory holds any number of case directories; a case directory
    holds ``case.yaml`` plus the input files it references.

    Parameters
    ----------
    root:
        Path to the goldsets directory (the parent of the step directories).
    step:
        If given, only load cases under that step directory.

    Raises
    ------
    EvalFixtureError
        On any validation failure (missing case.yaml, schema mismatch, step
        mismatch, missing input file, etc.).
    """
    cases: list[GoldCase] = []

    if not root.is_dir():
        raise EvalFixtureError("goldsets root is not a directory", root)

    for step_dir in sorted(root.iterdir()):
        if not step_dir.is_dir():
            continue

        step_name = step_dir.name
        if step and step_name != step:
            continue

        for case_dir in sorted(step_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            cases.append(_load_case(case_dir, step_name))

    # Deterministic ordering by (step, case_id)
    cases.sort(key=lambda c: (c.step, c.case_id))
    return cases


def _load_case(case_dir: Path, step_name: str) -> GoldCase:
    """Parse + validate one ``case_dir`` under the ``step_name`` directory."""
    case_yaml = case_dir / "case.yaml"
    if not case_yaml.exists():
        raise EvalFixtureError("missing case.yaml", case_dir)

    try:
        with open(case_yaml) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise EvalFixtureError(f"failed to parse case.yaml: {exc}", case_dir) from exc

    if not isinstance(data, dict):
        raise EvalFixtureError("case.yaml must contain a mapping", case_dir)

    # --- schema_version ---
    schema_version = data.get("schema_version")
    if schema_version is None:
        raise EvalFixtureError("missing schema_version", case_dir)
    if not isinstance(schema_version, int):
        raise EvalFixtureError(
            f"schema_version must be an int (got {type(schema_version).__name__})",
            case_dir,
        )
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise EvalFixtureError(
            f"unsupported schema_version {schema_version} (supported: {SUPPORTED_SCHEMA_VERSION})",
            case_dir,
        )

    # --- step must match the parent step directory ---
    case_step = data.get("step")
    if not case_step:
        raise EvalFixtureError("missing step field", case_dir)
    if case_step != step_name:
        raise EvalFixtureError(
            f"step field '{case_step}' does not match step directory '{step_name}'",
            case_dir,
        )

    # --- inputs validation ---
    raw_inputs: dict[str, str] = data.get("inputs", {}) or {}
    for logical_name, filename in raw_inputs.items():
        input_path = case_dir / filename
        if not input_path.exists():
            raise EvalFixtureError(
                f"input file '{filename}' (logical: '{logical_name}') not found",
                case_dir,
            )

    return GoldCase(
        step=case_step,  # type: ignore[arg-type]
        case_id=case_dir.name,
        case_dir=case_dir,
        schema_version=schema_version,
        holdout=bool(data.get("holdout", False)),
        inputs=raw_inputs,
        expected=data.get("expected", {}) or {},
        notes=data.get("notes", "") or "",
    )


def read_input(case: GoldCase, name: str) -> str:
    """Resolve and read a logical input file from *case*'s directory.

    Parameters
    ----------
    case:
        A validated :class:`GoldCase`.
    name:
        The logical input name (key in :attr:`GoldCase.inputs`).

    Returns
    -------
    str
        The decoded text content of the input file.

    Raises
    ------
    EvalFixtureError
        If *name* is not in the case's inputs mapping or the file cannot be read.
    """
    filename = case.inputs.get(name)
    if filename is None:
        raise EvalFixtureError(
            f"input name '{name}' not found in case inputs {list(case.inputs.keys())}",
            case.case_dir,
        )
    input_path = case.case_dir / filename
    try:
        return input_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvalFixtureError(f"failed to read input '{filename}': {exc}", case.case_dir) from exc
