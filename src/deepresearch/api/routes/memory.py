"""POST /memory/query."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from deepresearch.schemas.memory import MemoryRecord, MemoryType

router = APIRouter(prefix="/memory")


class MemoryQueryRequest(BaseModel):
    user_id: str
    project_id: str
    query: str
    memory_type: MemoryType = MemoryType.task
    top_k: int = 5


class MemoryQueryResponse(BaseModel):
    records: list[MemoryRecord]


@router.post("/query", response_model=MemoryQueryResponse)
async def memory_query(body: MemoryQueryRequest, request: Request) -> MemoryQueryResponse:
    deps = request.app.state.deps
    records = await deps.memory.query(
        run_id=None,
        user_id=body.user_id,
        project_id=body.project_id,
        query=body.query,
        memory_type=body.memory_type,
        top_k=body.top_k,
    )
    return MemoryQueryResponse(records=records)
