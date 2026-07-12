"""pick_disjoint is pure and conservative: mispicks waste work, never correctness."""

from __future__ import annotations

from forge.coding_pipeline.scheduling import pick_disjoint, scopes_overlap


def test_disjoint_scopes_all_batch():
    batch, deferred = pick_disjoint(
        [
            ("a", ["forge/task_worker/main.py"]),
            ("b", ["forge/coding_pipeline/dispatch.py"]),
            ("c", ["docs/"]),
        ]
    )
    assert batch == ["a", "b", "c"]
    assert deferred == []


def test_same_file_defers_the_later_leaf():
    batch, deferred = pick_disjoint(
        [("a", ["forge/x.py"]), ("b", ["forge/x.py"]), ("c", ["forge/y.py"])]
    )
    assert batch == ["a", "c"]  # order preserved; earlier leaf wins the contested scope
    assert deferred == ["b"]


def test_dir_vs_file_prefix_overlap_defers():
    batch, deferred = pick_disjoint([("a", ["forge/shared/"]), ("b", ["forge/shared/pool.py"])])
    assert batch == ["a"]
    assert deferred == ["b"]
    # and the other direction
    batch, deferred = pick_disjoint([("a", ["forge/shared/pool.py"]), ("b", ["forge/shared"])])
    assert batch == ["a"]
    assert deferred == ["b"]


def test_sibling_files_in_one_dir_do_not_overlap():
    assert not scopes_overlap(["forge/a.py"], ["forge/b.py"])


def test_empty_scope_defers_against_everything_but_batches_alone():
    # not first: defers
    batch, deferred = pick_disjoint([("a", ["forge/x.py"]), ("fixup", [])])
    assert batch == ["a"]
    assert deferred == ["fixup"]
    # first: dispatches alone, everything else defers
    batch, deferred = pick_disjoint([("fixup", []), ("a", ["forge/x.py"])])
    assert batch == ["fixup"]
    assert deferred == ["a"]


def test_nonempty_input_never_yields_empty_batch():
    batch, deferred = pick_disjoint([("only", [])])
    assert batch == ["only"]
    assert deferred == []


def test_multi_entry_scopes_overlap_on_any_entry():
    a = ["forge/x.py", "forge/shared/"]
    b = ["docs/readme.md", "forge/shared/pool.py"]
    assert scopes_overlap(a, b)
    batch, deferred = pick_disjoint([("a", a), ("b", b)])
    assert (batch, deferred) == (["a"], ["b"])


def test_degenerate_entry_claims_everything():
    assert scopes_overlap(["/"], ["forge/x.py"])
    assert scopes_overlap([""], ["forge/x.py"])
