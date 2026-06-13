"""Find today's Daily Note, check idempotency, and append code review blocks."""

from __future__ import annotations

import time

import httpx

from agents.code_reviewer.config import settings
from agents.code_reviewer.models import NightlyReport
from agents.code_reviewer.renderer import render_blocks
from agents.shared.models.nous import EditorJsBlock
from agents.shared.nous_http import nous_headers


def _base_url() -> str:
    return f"{settings.daemon_url}/api/notebooks/{settings.notebook_id}"


def _get_daily_note(date_str: str) -> dict | None:
    """Fetch today's daily note. Returns None on 404."""
    r = httpx.get(f"{_base_url()}/daily-notes/{date_str}", headers=nous_headers())
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()["data"]


def _get_page(page_id: str) -> dict | None:
    """Fetch a page by ID. Returns None on 404."""
    r = httpx.get(f"{_base_url()}/pages/{page_id}", headers=nous_headers())
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()["data"]


def _append_blocks(page_id: str, blocks: list[dict]) -> dict:
    """Append blocks to a page. Raises on failure."""
    r = httpx.post(
        f"{_base_url()}/pages/{page_id}/append",
        json={"blocks": blocks},
        headers=nous_headers(),
    )
    r.raise_for_status()
    return r.json()["data"]


def _wait_for_daily_note(date_str: str) -> dict | None:
    """Wait for the daily note to appear via daemon API."""
    for attempt in range(settings.find_attempts):
        page = _get_daily_note(date_str)
        if page is not None:
            return page
        if attempt < settings.find_attempts - 1:
            print(
                f"Daily note '{date_str}' not found, "
                f"retrying in {settings.find_delay_seconds}s "
                f"(attempt {attempt + 1}/{settings.find_attempts})"
            )
            time.sleep(settings.find_delay_seconds)

    print(f"Daily note '{date_str}' not found after {settings.find_attempts} attempts")
    return None


def _has_review(page_id: str) -> bool:
    """Check if the daily note already has a code review (idempotency)."""
    page = _get_page(page_id)
    if page is None:
        return False

    blocks = page.get("content", {}).get("blocks", [])
    for block in blocks:
        text = block.get("data", {}).get("text", "")
        if settings.review_marker in text:
            return True

    return False


def append_to_daily_note(report: NightlyReport) -> dict | None:
    """Append code review to today's Daily Note via the daemon API.

    Returns the append response dict, or None if the note is missing
    or the review was already present.
    """
    # Wait for the daily note to exist
    page = _wait_for_daily_note(report.date)
    if page is None:
        return None

    page_id = page["id"]

    # Idempotency check
    if _has_review(page_id):
        print(f"Daily note '{report.date}' already has a code review, skipping")
        return None

    # Render and append
    blocks = render_blocks(report)
    result = _append_blocks(page_id, [b.model_dump() for b in blocks])
    print(f"Code review appended ({result.get('blocksAdded', '?')} blocks)")
    return result
