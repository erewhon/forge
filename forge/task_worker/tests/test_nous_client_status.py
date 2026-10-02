"""Row cells written by update_task_status, without a daemon."""

from __future__ import annotations

from datetime import date

from forge.task_worker.nous_client import _status_cells

_WITH_COLUMN = {"properties": [{"name": "Status"}, {"name": "implemented by"}]}
_WITHOUT_COLUMN = {"properties": [{"name": "Status"}, {"name": "Completed"}]}


def test_done_stamps_implemented_by_when_the_column_exists() -> None:
    cells = _status_cells("Done", _WITH_COLUMN)
    assert cells == {
        "Status": "Done",
        "Completed": date.today().isoformat(),
        "Implemented By": "worker",
    }


def test_done_skips_the_stamp_when_the_column_is_missing() -> None:
    cells = _status_cells("Done", _WITHOUT_COLUMN)
    assert cells == {"Status": "Done", "Completed": date.today().isoformat()}


def test_other_statuses_never_stamp() -> None:
    for status in ("Ready", "In Progress", "Spec Needed"):
        cells = _status_cells(status, _WITH_COLUMN, execution_mode="Manual")
        assert cells == {"Status": status, "Execution Mode": "Manual"}
