"""TraceRecorder — single sink that AgentStep, MemoryEvent, and
ModelCallRecord events flow through during a run. Writes them to SQLite via
the storage repos."""

from __future__ import annotations

from dataclasses import dataclass

from deepresearch.schemas.agents import AgentStep
from deepresearch.schemas.memory import MemoryEvent
from deepresearch.schemas.models import ModelCallRecord
from deepresearch.storage.repository import Repositories


@dataclass
class TraceRecorder:
    repos: Repositories

    def record_step(self, step: AgentStep) -> None:
        self.repos.steps.append(step)

    def record_memory_event(self, event: MemoryEvent) -> None:
        self.repos.memory_events.append(event)

    def record_model_call(self, rec: ModelCallRecord) -> None:
        self.repos.model_calls.append(rec)
