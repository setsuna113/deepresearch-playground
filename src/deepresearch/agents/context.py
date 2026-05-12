"""RunContext — shared mutable state for one research run.

The orchestrator owns this and threads it through every agent. Agents
mutate the marked-mutable fields and return AgentSteps; everything that
needs to be persisted goes through `ctx.deps.recorder`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deepresearch.config.schema import AppConfig
from deepresearch.memory.profiles import MemoryProfile
from deepresearch.memory.service import MemoryService
from deepresearch.models.client import ModelClient
from deepresearch.models.router import Router
from deepresearch.observability.trace import TraceRecorder
from deepresearch.schemas.agents import ReflectionUpdate
from deepresearch.schemas.memory import MemoryRecord, MemoryType
from deepresearch.schemas.runs import ResearchRun, RunRequest
from deepresearch.schemas.search import Citation, Evidence, SearchDocument, SubQuestion
from deepresearch.storage.repository import Repositories
from deepresearch.tools.registry import ToolRegistry


@dataclass
class RunDependencies:
    config: AppConfig
    repos: Repositories
    recorder: TraceRecorder
    model_client: ModelClient
    router: Router
    memory: MemoryService
    tools: ToolRegistry


@dataclass
class RunContext:
    run: ResearchRun
    request: RunRequest
    deps: RunDependencies
    memory_profile: MemoryProfile

    # Mutable scratch
    primes: dict[MemoryType, list[MemoryRecord]] = field(default_factory=dict)
    plan: list[SubQuestion] = field(default_factory=list)
    documents: list[SearchDocument] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    draft_report: str | None = None
    citations: list[Citation] = field(default_factory=list)
    reflection: ReflectionUpdate | None = None
    _step_seq: int = 0

    def next_seq(self) -> int:
        self._step_seq += 1
        return self._step_seq
