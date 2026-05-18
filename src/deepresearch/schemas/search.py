"""Search + document + evidence + citation shapes."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    id: str
    text: str
    rationale: str | None = None


class SearchQuery(BaseModel):
    subquestion_id: str
    query: str
    provider: str = "tavily"


class SearchDocument(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    subquestion_id: str
    url: str
    title: str | None = None
    snippet: str | None = None
    content_md: str | None = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_provider: str = "tavily"
    score: float | None = None


class Evidence(BaseModel):
    document_id: UUID
    subquestion_id: str
    quote: str
    url: str
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)


class Citation(BaseModel):
    marker: str  # e.g., "[1]"
    url: str
    title: str | None = None
    quote: str | None = None
