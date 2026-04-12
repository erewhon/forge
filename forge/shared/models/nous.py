"""Pydantic models for Nous data structures shared across agents.

Page creation is handled by nous_ai.page_storage.NousPageStorage.
These models cover blocks (used by the renderer) and folders (not yet
managed by NousPageStorage).
"""

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field


def _new_id() -> str:
    return str(uuid4())


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f000Z")


class EditorJsBlock(BaseModel):
    id: str = Field(default_factory=_new_id)
    type: str
    data: dict


class NousFolder(BaseModel):
    id: str = Field(default_factory=_new_id)
    notebookId: str
    name: str
    sectionId: str | None = None
    folderType: str = "standard"
    parentId: str | None = None
    isArchived: bool = False
    position: int = 100
    createdAt: str = Field(default_factory=_now_iso)
    updatedAt: str = Field(default_factory=_now_iso)
