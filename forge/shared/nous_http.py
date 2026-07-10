"""Shared Nous daemon HTTP helpers.

Auto-discovers the API key from ~/.local/share/nous/daemon-api-key (same
pattern as nous_mcp.daemon_client). Falls back to NOUS_API_KEY env var.

Usage:
    from forge.shared.nous_http import nous_headers, nous_auth_kwargs

    r = httpx.get(url, headers=nous_headers())
    # or for httpx.Client:
    with httpx.Client(**nous_auth_kwargs()) as client:
        ...
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

DEFAULT_KEY_FILE = Path.home() / ".local" / "share" / "nous" / "daemon-api-key"


@lru_cache(maxsize=1)
def discover_api_key(key_file: Path = DEFAULT_KEY_FILE) -> str | None:
    """Find the first rw: key in the daemon key file, or NOUS_API_KEY env var."""
    env_key = os.environ.get("NOUS_API_KEY")
    if env_key:
        return env_key
    if not key_file.exists():
        return None
    try:
        for line in key_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and line.startswith("rw:"):
                return line
    except OSError:
        return None
    return None


def nous_headers() -> dict[str, str]:
    """Return headers dict with Authorization: Bearer <key> if a key is available."""
    key = discover_api_key()
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}"}


def nous_auth_kwargs() -> dict[str, object]:
    """Return kwargs for httpx.Client(...) that include auth headers."""
    return {"headers": nous_headers()}
