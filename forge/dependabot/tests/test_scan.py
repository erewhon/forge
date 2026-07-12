from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from forge.dependabot.config import settings
from forge.dependabot.scan import ScanError, classify_delta, scan_outdated

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "uv_tree_outdated.txt"


def _make_process(returncode: int, stdout: str = "", stderr: str = ""):
    """Build a CompletedProcess suitable for patching subprocess.run."""
    return CompletedProcess(
        args=["uv", "tree"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


# ---------------------------------------------------------------------------
# classify_delta
# ---------------------------------------------------------------------------


class TestClassifyDelta:
    def test_patch_delta(self):
        assert classify_delta("1.2.3", "1.2.4") == "patch"

    def test_minor_delta(self):
        assert classify_delta("1.2.3", "1.3.0") == "minor"

    def test_major_delta(self):
        assert classify_delta("1.2.3", "2.0.0") == "major"

    def test_unknown_non_integer(self):
        assert classify_delta("1.2.3", "2024.1") == "unknown"

    def test_patch_with_v_prefix(self):
        assert classify_delta("v1.2.3", "v1.2.4") == "patch"

    def test_minor_with_v_prefix(self):
        assert classify_delta("v0.79.0", "v0.116.0") == "minor"

    def test_identical_versions(self):
        assert classify_delta("1.0.0", "1.0.0") == "patch"

    def test_two_segment_diff_major(self):
        assert classify_delta("1.0", "2.0") == "major"

    def test_two_segment_diff_minor(self):
        assert classify_delta("1.0", "1.1") == "minor"


# ---------------------------------------------------------------------------
# scan_outdated — fixture parsing
# ---------------------------------------------------------------------------


class TestScanOutdated:
    def test_fixture_parsed_into_candidates(self):
        """Parser turns the fixture into the expected BumpCandidate list."""
        fixture_text = FIXTURE_PATH.read_text()
        mock = patch("forge.dependabot.scan.subprocess.run")
        m = mock.start()
        m.return_value = _make_process(0, fixture_text, "")

        candidates = scan_outdated(Path("/fake/repo"))

        mock.stop()

        # Only lines with (latest: ...) should be captured
        names = [c.name for c in candidates]
        assert "anthropic" in names
        assert "beautifulsoup4" in names
        assert "mcp" in names
        assert "openai" in names
        assert "pydantic" in names
        assert "pydantic-settings" in names
        assert "typer" in names
        assert "pytest" in names
        # Lines without (latest: ...) are skipped
        assert "httpx" not in names
        assert "nous-ai" not in names
        assert "pyyaml" not in names

    def test_assert_first_entries(self):
        """Assert exact first entries — patch first, then alphabetical."""
        fixture_text = FIXTURE_PATH.read_text()
        mock = patch("forge.dependabot.scan.subprocess.run")
        m = mock.start()
        m.return_value = _make_process(0, fixture_text, "")

        candidates = scan_outdated(Path("/fake/repo"))

        mock.stop()

        # Verify sorting: patch first
        patch_first = candidates[0]
        assert patch_first.delta == "patch"

    def test_candidate_fields_correct(self):
        """Each BumpCandidate has correct name, current, latest, delta."""
        fixture_text = FIXTURE_PATH.read_text()
        mock = patch("forge.dependabot.scan.subprocess.run")
        m = mock.start()
        m.return_value = _make_process(0, fixture_text, "")

        candidates = scan_outdated(Path("/fake/repo"))

        mock.stop()

        # Find mcp — it has a patch delta
        mcp = next(c for c in candidates if c.name == "mcp")
        assert mcp.current == "1.28.0"
        assert mcp.latest == "1.28.1"
        assert mcp.delta == "patch"

    def test_non_zero_exit_raises_scan_error(self):
        """Non-zero uv exit raises ScanError with stderr message."""
        mock = patch("forge.dependabot.scan.subprocess.run")
        m = mock.start()
        m.return_value = _make_process(1, "", "error: could not find pyproject.toml")

        with pytest.raises(ScanError) as exc_info:
            scan_outdated(Path("/fake/repo"))

        mock.stop()
        assert "could not find pyproject.toml" in str(exc_info.value)

    def test_empty_stdout_returns_empty_list(self):
        """Empty uv output returns an empty list (not an error)."""
        mock = patch("forge.dependabot.scan.subprocess.run")
        m = mock.start()
        m.return_value = _make_process(0, "", "")

        candidates = scan_outdated(Path("/fake/repo"))

        mock.stop()
        assert candidates == []

    def test_max_candidates_capped(self):
        """Results are capped at settings.max_candidates."""
        fixture_text = FIXTURE_PATH.read_text()
        mock = patch("forge.dependabot.scan.subprocess.run")
        m = mock.start()
        m.return_value = _make_process(0, fixture_text, "")

        # Temporarily override settings
        original = settings.max_candidates
        settings.max_candidates = 2

        try:
            candidates = scan_outdated(Path("/fake/repo"))
            assert len(candidates) <= 2
        finally:
            settings.max_candidates = original

        mock.stop()

    def test_sorting_patch_before_minor(self):
        """Verify patch delta entries come before minor ones."""
        fixture_text = FIXTURE_PATH.read_text()
        mock = patch("forge.dependabot.scan.subprocess.run")
        m = mock.start()
        m.return_value = _make_process(0, fixture_text, "")

        candidates = scan_outdated(Path("/fake/repo"))

        mock.stop()

        deltas = [c.delta for c in candidates]
        # Find the last patch and first minor
        try:
            last_patch_idx = max(i for i, d in enumerate(deltas) if d == "patch")
            first_minor_idx = min(i for i, d in enumerate(deltas) if d == "minor")
            assert last_patch_idx < first_minor_idx
        except ValueError:
            # Only one class present — that's fine
            pass

    def test_sorting_alphabetical_within_class(self):
        """Entries with the same delta are sorted alphabetically."""
        fixture_text = FIXTURE_PATH.read_text()
        mock = patch("forge.dependabot.scan.subprocess.run")
        m = mock.start()
        m.return_value = _make_process(0, fixture_text, "")

        candidates = scan_outdated(Path("/fake/repo"))

        mock.stop()

        # Group by delta
        for delta in ("patch", "minor"):
            items = [c for c in candidates if c.delta == delta]
            names = [c.name for c in items]
            assert names == sorted(names), f"Not sorted alphabetically within {delta}"

    def test_calls_uv_with_correct_args(self):
        """Verify the correct uv command arguments are used."""
        mock = patch("forge.dependabot.scan.subprocess.run")
        m = mock.start()
        m.return_value = _make_process(0, "", "")

        scan_outdated(Path("/fake/repo"))

        mock.stop()
        m.assert_called_once_with(
            # --frozen: the scan must never rewrite a stale uv.lock (fleet-clone finding).
            ["uv", "tree", "--outdated", "--depth", "1", "--no-dedupe", "--frozen"],
            cwd="/fake/repo",
            capture_output=True,
            text=True,
            timeout=settings.scan_timeout,
        )

    def test_custom_timeout_passed_through(self):
        """Custom timeout overrides settings.scan_timeout."""
        mock = patch("forge.dependabot.scan.subprocess.run")
        m = mock.start()
        m.return_value = _make_process(0, "", "")

        scan_outdated(Path("/fake/repo"), timeout=60)

        mock.stop()
        call_kwargs = m.call_args
        assert call_kwargs.kwargs["timeout"] == 60
