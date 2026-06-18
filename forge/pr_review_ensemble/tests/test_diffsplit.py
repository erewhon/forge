"""Unit tests for the diff splitter (pure, no LLM)."""

from __future__ import annotations

from agents.pr_review_ensemble.diffsplit import split_diff


def _file_diff(path: str, *, hunks: int = 1, lines_per_hunk: int = 1) -> str:
    s = f"diff --git a/{path} b/{path}\nindex 0..1 100644\n--- a/{path}\n+++ b/{path}\n"
    for h in range(hunks):
        s += f"@@ -{h + 1},1 +{h + 1},{lines_per_hunk} @@\n"
        s += "".join(f"+line {h}-{i}\n" for i in range(lines_per_hunk))
    return s


def test_single_file_one_chunk():
    diff = _file_diff("src/a.py")
    chunks = split_diff(diff, chunk_chars=10_000)
    assert len(chunks) == 1
    assert chunks[0].files == ["src/a.py"]
    assert "src/a.py" in chunks[0].text


def test_packs_small_files_into_one_chunk():
    diff = _file_diff("a.py") + _file_diff("b.py") + _file_diff("c.py")
    chunks = split_diff(diff, chunk_chars=10_000)
    assert len(chunks) == 1
    assert chunks[0].files == ["a.py", "b.py", "c.py"]


def test_separate_chunks_when_budget_tight():
    a, b = _file_diff("a.py"), _file_diff("b.py")
    # budget fits one file but not two
    chunks = split_diff(a + b, chunk_chars=len(a) + 5)
    assert len(chunks) == 2
    assert [c.files for c in chunks] == [["a.py"], ["b.py"]]


def test_reassembles_to_original():
    diff = _file_diff("a.py") + _file_diff("b.py") + _file_diff("c.py")
    chunks = split_diff(diff, chunk_chars=len(_file_diff("a.py")) + 5)
    assert "".join(c.text for c in chunks) == diff  # no bytes lost when packing


def test_splits_oversized_file_by_hunk():
    diff = _file_diff("big.py", hunks=3)
    one_hunk = _file_diff("big.py", hunks=1)
    # budget holds the header + ~1 hunk but not the whole 3-hunk file
    chunks = split_diff(diff, chunk_chars=len(one_hunk) + 5)
    assert len(chunks) > 1
    assert all(c.files == ["big.py"] for c in chunks)
    # each piece keeps the file header so it is self-describing
    assert all("diff --git a/big.py" in c.text for c in chunks)


def test_truncates_single_giant_hunk():
    diff = _file_diff("huge.py", hunks=1, lines_per_hunk=500)
    chunks = split_diff(diff, chunk_chars=400)
    assert len(chunks) == 1
    assert "truncated" in chunks[0].text
    assert len(chunks[0].text) <= 400 + 80  # header + truncated marker slack


def test_no_git_header_fallback():
    plain = "--- a/x\n+++ b/x\n@@ -1 +1 @@\n+hi\n"
    chunks = split_diff(plain, chunk_chars=10_000)
    assert len(chunks) == 1
    assert chunks[0].files == []
    assert chunks[0].text == plain


def test_empty_diff():
    assert split_diff("", chunk_chars=1000) == []
