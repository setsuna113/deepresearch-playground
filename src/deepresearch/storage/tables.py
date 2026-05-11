"""SQLModel tables. JSON columns hold nested pydantic blobs to avoid
premature normalization; we trade query power for schema agility because
later phases will reshape these fields several times."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.utcnow()


class ResearchRunTable(SQLModel, table=True):
    __tablename__ = "research_runs"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: str = Field(index=True)
    project_id: str = Field(index=True)
    query: str
    depth: str
    max_searches: int
    model_profile: str
    memory_profile: str
    status: str = Field(index=True)
    created_at: datetime = Field(default_factory=utcnow, index=True)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    report_md: str | None = None
    citations_json: str | None = None
    error: str | None = None
    privacy_envelope_json: str
    metrics_json: str | None = None


class AgentStepTable(SQLModel, table=True):
    __tablename__ = "agent_steps"
    __table_args__ = (UniqueConstraint("run_id", "seq"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    run_id: UUID = Field(foreign_key="research_runs.id", index=True)
    seq: int
    role: str
    status: str
    input_json: str
    output_json: str
    started_at: datetime
    finished_at: datetime | None = None
    latency_ms: int = 0
    model_call_ids_json: str | None = None
    error: str | None = None


class SearchDocumentTable(SQLModel, table=True):
    __tablename__ = "search_documents"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    run_id: UUID = Field(foreign_key="research_runs.id", index=True)
    subquestion_id: str
    url: str = Field(index=True)
    title: str | None = None
    snippet: str | None = None
    content_md: str | None = None
    fetched_at: datetime = Field(default_factory=utcnow)
    source_provider: str
    score: float | None = None
    extracted_evidence_json: str | None = None


class MemoryEventTable(SQLModel, table=True):
    __tablename__ = "memory_events"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    run_id: UUID | None = Field(default=None, foreign_key="research_runs.id", index=True)
    user_id: str = Field(index=True)
    project_id: str = Field(index=True)
    kind: str
    memory_type: str
    query: str | None = None
    payload_json: str
    reme_ids_json: str | None = None
    backend: str = "reme"
    at: datetime = Field(default_factory=utcnow)


class ModelCallRecordTable(SQLModel, table=True):
    __tablename__ = "model_calls"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    run_id: UUID | None = Field(default=None, foreign_key="research_runs.id", index=True)
    step_id: UUID | None = Field(default=None, foreign_key="agent_steps.id", index=True)
    endpoint_name: str
    model_id: str
    role: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    started_at: datetime = Field(default_factory=utcnow)
    privacy_envelope_json: str
    request_hash: str | None = None
    error: str | None = None
