"""POST /research_runs, GET /research_runs/{id}, GET /research_runs/{id}/trace."""

from __future__ import annotations

import asyncio
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request

from deepresearch.agents.orchestrator import run_research
from deepresearch.schemas.runs import ResearchRun, RunRequest, RunResponse, RunStatus
from deepresearch.schemas.trace import Trace

router = APIRouter(prefix="/research_runs")


@router.post("", response_model=RunResponse)
async def create_run(req: RunRequest, request: Request) -> RunResponse:
    deps = request.app.state.deps
    # Pre-create the row so the caller gets a stable run_id immediately,
    # then run the orchestrator in the background against the same row.
    run = ResearchRun(request=req, status=RunStatus.pending)
    deps.repos.runs.create(run)
    # Hold a reference so asyncio doesn't garbage-collect the task
    # mid-run; the FastAPI app's state outlives the request.
    task = asyncio.create_task(run_research(req, deps, existing_run=run))
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return RunResponse(run_id=run.id, status=RunStatus.pending)


_BACKGROUND_TASKS: set[asyncio.Task] = set()


@router.get("/{run_id}", response_model=RunResponse)
async def get_run(run_id: UUID, request: Request) -> RunResponse:
    deps = request.app.state.deps
    run = deps.repos.runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return RunResponse(
        run_id=run.id,
        status=run.status,
        report_md=run.report_md,
        citations=run.citations,
        metrics=run.metrics,
        error=run.error,
    )


@router.get("/{run_id}/trace", response_model=Trace)
async def get_trace(run_id: UUID, request: Request) -> Trace:
    deps = request.app.state.deps
    run = deps.repos.runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    steps = deps.repos.steps.list_for_run(run_id)
    memory_events = deps.repos.memory_events.list_for_run(run_id)
    model_calls = deps.repos.model_calls.list_for_run(run_id)
    return Trace(run=run, steps=steps, memory_events=memory_events, model_calls=model_calls)
