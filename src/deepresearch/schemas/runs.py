"""Top-level run shapes. ResearchRun is persisted; RunRequest/RunResponse
are the public API contract."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from deepresearch.schemas.privacy import PrivacyEnvelope
from deepresearch.schemas.search import Citation


class Depth(str, Enum):
    quick = "quick"
    standard = "standard"
    deep = "deep"


class RunStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"


class RunRequest(BaseModel):
    query: str
    user_id: str
    project_id: str
    depth: Depth = Depth.standard
    max_searches: int = Field(default=5, ge=1, le=50)
    max_concurrent_units: int = Field(default=2, ge=1, le=8)
    model_profile: str = "phase1_default"
    memory_profile: str = "default"
    privacy_envelope: PrivacyEnvelope = Field(default_factory=PrivacyEnvelope.default_public)


class RunMetrics(BaseModel):
    total_latency_ms: int = 0
    n_searches: int = 0
    n_documents: int = 0
    n_prompt_tokens: int = 0
    n_completion_tokens: int = 0
    n_memory_reads: int = 0
    n_memory_writes: int = 0


class ResearchRun(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    request: RunRequest
    status: RunStatus = RunStatus.pending
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    report_md: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    error: str | None = None
    metrics: RunMetrics = Field(default_factory=RunMetrics)


class RunResponse(BaseModel):
    run_id: UUID
    status: RunStatus
    report_md: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    metrics: RunMetrics | None = None
    error: str | None = None
