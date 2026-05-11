"""Agent ABC.

Every agent reads + mutates the shared RunContext, makes (typically one)
LLM call, and returns an AgentStep describing what it did. The orchestrator
is the only place the sequence is encoded.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

from deepresearch.schemas.agents import AgentRole, AgentStep, StepStatus

if TYPE_CHECKING:
    from deepresearch.agents.context import RunContext


class Agent(ABC):
    role: AgentRole

    @abstractmethod
    async def _run(self, ctx: "RunContext") -> dict:
        """Implementation hook. Return the step's output dict."""

    async def run(self, ctx: "RunContext", seq: int) -> AgentStep:
        started = datetime.utcnow()
        t0 = time.perf_counter()
        step = AgentStep(
            run_id=ctx.run.id, seq=seq, role=self.role, started_at=started, input={}
        )
        try:
            output = await self._run(ctx)
            step.output = output
            step.status = StepStatus.ok
        except Exception as e:
            step.status = StepStatus.error
            step.error = repr(e)
            raise
        finally:
            step.finished_at = datetime.utcnow()
            step.latency_ms = int((time.perf_counter() - t0) * 1000)
            if ctx.deps.recorder is not None:
                ctx.deps.recorder.record_step(step)
        return step
