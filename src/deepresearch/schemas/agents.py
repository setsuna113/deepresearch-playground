"""AgentStep + role enum + structured reflection output.

`BroadcastCandidate` is reserved for Phase 5 (Reflection Broadcast Protocol);
in Phase 1 the Reflector may emit it but no consumer reads it yet.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    # Legacy STORM roles (Phase 1, pre-LangGraph). Kept until commit 3
    # deletes the custom agents that emit them; existing SQLite rows
    # continue to deserialize.
    planner = "planner"
    searcher = "searcher"
    reader = "reader"
    synthesizer = "synthesizer"
    # LangGraph roles (Phase 1.5 onwards).
    supervisor = "supervisor"
    researcher = "researcher"
    compressor = "compressor"
    final_report = "final_report"
    # Reflector spans both eras.
    reflector = "reflector"


class StepStatus(str, Enum):
    ok = "ok"
    error = "error"
    skipped = "skipped"


class AgentStep(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    run_id: UUID | None = None
    seq: int
    role: AgentRole
    status: StepStatus = StepStatus.ok
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: datetime | None = None
    latency_ms: int = 0
    model_call_ids: list[UUID] = Field(default_factory=list)
    error: str | None = None


class BroadcastCandidate(BaseModel):
    """A reflection candidate that *may* be broadcast laterally to cloud
    subagents under the mutual-information budget (Phase 5)."""

    summary: str
    target_audiences: list[str] = Field(default_factory=list)  # e.g., ["local", "cloud"]
    estimated_bits: float = 0.0
    requires_ci_filter: bool = True


class ReflectionUpdate(BaseModel):
    personal_update: str | None = None
    task_update: str | None = None
    tool_update: str | None = None
    needs_revision: bool = False
    broadcast_candidate: BroadcastCandidate | None = None
