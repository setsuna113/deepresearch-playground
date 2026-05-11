"""LLM endpoint + model-call record shapes."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from deepresearch.schemas.privacy import PrivacyEnvelope


class Endpoint(BaseModel):
    name: str  # "local" | "cloud" | "judge" | ...
    base_url: str
    api_key: str = "EMPTY"
    model_id: str
    role_hint: str | None = None


class ModelProfile(BaseModel):
    """Maps agent role -> endpoint name."""

    name: str
    planner: str
    searcher: str
    reader: str
    synthesizer: str
    reflector: str


class ModelCallRecord(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    run_id: UUID | None = None
    step_id: UUID | None = None
    endpoint_name: str
    model_id: str
    role: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    started_at: datetime = Field(default_factory=datetime.utcnow)
    envelope: PrivacyEnvelope = Field(default_factory=PrivacyEnvelope.default_public)
    request_hash: str | None = None
    error: str | None = None
