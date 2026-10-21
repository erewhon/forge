from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from forge.fanout.profiles import Profile

SeatKind = Literal["claude", "pi"]
SeatStatus = Literal["ok", "empty", "timeout", "error", "cancelled", "pending"]


class SeatSpec(BaseModel):
    """One seat to run: which agent CLI (kind) drives which model.

    claude -> `claude -p` with a Claude model id or alias (opus, fable, claude-opus-4-8, ...).
    pi     -> `pi -p --provider <provider> --model <id>` over the LLM router (local fleet).
    `display` is the label shown in reports and used in findings file names.
    """

    label: str  # A, B, ...
    kind: SeatKind
    model: str
    display: str


class SeatRun(BaseModel):
    label: str
    kind: SeatKind
    model: str
    display: str
    status: SeatStatus
    cmd: list[str]
    findings_path: Path | None = None
    findings_chars: int = 0
    latency_ms: int | None = None
    returncode: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    error_message: str | None = None


class SweepManifest(BaseModel):
    """Everything the judge (and a human) needs to know about one sweep, as DIR/manifest.json."""

    target: Path  # the directory the seats ran in (a repo, a folder of papers, ...)
    out_dir: Path
    profile: Profile  # resolved at sweep time, so `judge` never needs the profile file again
    timestamp: datetime
    user_prompt_path: Path  # the prompt as the user wrote it
    prompt_path: Path  # the wrapped prompt as sent to every seat
    seats: list[SeatRun]
