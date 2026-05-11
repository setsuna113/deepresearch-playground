"""Storage round-trip: create a run, append a step, read it back."""

from __future__ import annotations

from datetime import datetime

from deepresearch.config import get_config
from deepresearch.schemas.agents import AgentRole, AgentStep, StepStatus
from deepresearch.schemas.memory import MemoryEvent, MemoryEventKind, MemoryType
from deepresearch.schemas.runs import Depth, ResearchRun, RunRequest, RunStatus
from deepresearch.storage.db import init_db
from deepresearch.storage.repository import Repositories


def test_run_step_memory_roundtrip():
    cfg = get_config()
    engine = init_db(cfg.app.sqlite_path)
    repos = Repositories.from_engine(engine)

    req = RunRequest(query="q", user_id="u", project_id="p", depth=Depth.quick)
    run = ResearchRun(request=req, status=RunStatus.pending)
    repos.runs.create(run)
    repos.runs.mark_running(run.id)

    step = AgentStep(
        run_id=run.id,
        seq=1,
        role=AgentRole.planner,
        status=StepStatus.ok,
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
        latency_ms=10,
    )
    repos.steps.append(step)
    steps = repos.steps.list_for_run(run.id)
    assert len(steps) == 1
    assert steps[0].role == AgentRole.planner

    event = MemoryEvent(
        run_id=run.id,
        user_id="u",
        project_id="p",
        kind=MemoryEventKind.prime_read,
        memory_type=MemoryType.task,
        query="q",
    )
    repos.memory_events.append(event)
    events = repos.memory_events.list_for_run(run.id)
    assert len(events) == 1
    assert events[0].kind == MemoryEventKind.prime_read


def test_run_get_after_done():
    cfg = get_config()
    engine = init_db(cfg.app.sqlite_path)
    repos = Repositories.from_engine(engine)
    req = RunRequest(query="q2", user_id="u", project_id="p", depth=Depth.quick)
    run = ResearchRun(request=req, status=RunStatus.pending)
    repos.runs.create(run)
    run.status = RunStatus.done
    run.report_md = "hello"
    run.finished_at = datetime.utcnow()
    repos.runs.mark_done(run)

    back = repos.runs.get(run.id)
    assert back is not None
    assert back.report_md == "hello"
    assert back.status == RunStatus.done
