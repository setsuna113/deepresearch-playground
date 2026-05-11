"""Aggregated trace view — joins steps + memory events + model calls by run_id."""

from __future__ import annotations

from pydantic import BaseModel, Field

from deepresearch.schemas.agents import AgentStep
from deepresearch.schemas.memory import MemoryEvent
from deepresearch.schemas.models import ModelCallRecord
from deepresearch.schemas.runs import ResearchRun


class Trace(BaseModel):
    run: ResearchRun
    steps: list[AgentStep] = Field(default_factory=list)
    memory_events: list[MemoryEvent] = Field(default_factory=list)
    model_calls: list[ModelCallRecord] = Field(default_factory=list)
