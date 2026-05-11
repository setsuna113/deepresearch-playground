"""MemoryType + MemoryEvent + MemoryRecord.

The public 4-type taxonomy (personal/task/tool/working) is used
throughout the agent layer. `memory/types.py` handles the mapping to ReMe's
native 3-type taxonomy where needed.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from deepresearch.schemas.privacy import PrivacyEnvelope


class MemoryType(str, Enum):
    personal = "personal"
    task = "task"
    tool = "tool"
    working = "working"


class MemoryEventKind(str, Enum):
    prime_read = "prime_read"
    adhoc_read = "adhoc_read"
    post_write = "post_write"
    working_write = "working_write"


class MemoryRecord(BaseModel):
    """A single retrieved or written memory."""

    id: str  # backend-native id (ReMe uses str)
    memory_type: MemoryType
    content: str
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    envelope: PrivacyEnvelope = Field(default_factory=PrivacyEnvelope.default_public)


class MemoryEvent(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    run_id: UUID | None = None
    user_id: str
    project_id: str
    kind: MemoryEventKind
    memory_type: MemoryType
    query: str | None = None
    records: list[MemoryRecord] = Field(default_factory=list)
    backend: str = "reme"  # "reme" | "working_qdrant"
    at: datetime = Field(default_factory=datetime.utcnow)
