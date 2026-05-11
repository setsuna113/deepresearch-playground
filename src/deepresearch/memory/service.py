"""MemoryService — the only memory surface the rest of the app sees.

Routes by memory type:
- personal / procedural / tool  -> ReMeAdapter
- working                       -> WorkingMemory (our Qdrant)

Records a MemoryEvent on every read/write via the TraceRecorder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from deepresearch.config.schema import MemorySection
from deepresearch.memory.profiles import MemoryProfile
from deepresearch.memory.reme_adapter import ReMeAdapter
from deepresearch.memory.working_memory import WorkingMemory
from deepresearch.schemas.memory import (
    MemoryEvent,
    MemoryEventKind,
    MemoryRecord,
    MemoryType,
)


class _RecorderProtocol(Protocol):
    def record_memory_event(self, event: MemoryEvent) -> None: ...


@dataclass
class MemoryService:
    reme: ReMeAdapter
    working: WorkingMemory
    recorder: _RecorderProtocol | None = None

    @classmethod
    async def create(
        cls,
        section: MemorySection,
        recorder: _RecorderProtocol | None = None,
    ) -> "MemoryService":
        reme = await ReMeAdapter.create(section.reme)
        working = await WorkingMemory.create(section.working)
        return cls(reme=reme, working=working, recorder=recorder)

    async def query(
        self,
        *,
        run_id: UUID | None,
        user_id: str,
        project_id: str,
        query: str,
        memory_type: MemoryType,
        top_k: int,
        kind: MemoryEventKind = MemoryEventKind.adhoc_read,
    ) -> list[MemoryRecord]:
        if memory_type == MemoryType.working:
            records = await self.working.query(
                user_id=user_id, project_id=project_id, query=query, top_k=top_k
            )
            backend = "working_qdrant"
        else:
            records = await self.reme.query(
                user_id=user_id,
                project_id=project_id,
                query=query,
                memory_type=memory_type,
                top_k=top_k,
            )
            backend = "reme"
        event = MemoryEvent(
            run_id=run_id,
            user_id=user_id,
            project_id=project_id,
            kind=kind,
            memory_type=memory_type,
            query=query,
            records=records,
            backend=backend,
        )
        if self.recorder is not None:
            self.recorder.record_memory_event(event)
        return records

    async def write(
        self,
        *,
        run_id: UUID | None,
        user_id: str,
        project_id: str,
        memory_type: MemoryType,
        content: str,
        metadata: dict | None = None,
        kind: MemoryEventKind = MemoryEventKind.post_write,
    ) -> MemoryRecord | None:
        if memory_type == MemoryType.working:
            rec = await self.working.write(
                user_id=user_id, project_id=project_id, content=content, metadata=metadata
            )
            backend = "working_qdrant"
        else:
            rec = await self.reme.write(
                user_id=user_id,
                project_id=project_id,
                memory_type=memory_type,
                content=content,
                metadata=metadata,
            )
            backend = "reme"
        if rec is None:
            return None
        event = MemoryEvent(
            run_id=run_id,
            user_id=user_id,
            project_id=project_id,
            kind=kind,
            memory_type=memory_type,
            query=None,
            records=[rec],
            backend=backend,
        )
        if self.recorder is not None:
            self.recorder.record_memory_event(event)
        return rec

    async def query_prime(
        self,
        *,
        run_id: UUID | None,
        user_id: str,
        project_id: str,
        query: str,
        profile: MemoryProfile,
    ) -> dict[MemoryType, list[MemoryRecord]]:
        """One-shot read used at the start of a research run.

        Reads personal / procedural / tool (and optionally working) and
        records each as a `prime_read` event.
        """
        result: dict[MemoryType, list[MemoryRecord]] = {}
        for mt in (MemoryType.personal, MemoryType.procedural, MemoryType.tool, MemoryType.working):
            k = profile.top_k_for(mt)
            result[mt] = await self.query(
                run_id=run_id,
                user_id=user_id,
                project_id=project_id,
                query=query,
                memory_type=mt,
                top_k=k,
                kind=MemoryEventKind.prime_read,
            )
        return result
