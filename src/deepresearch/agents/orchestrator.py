"""Orchestrator — the only place the STORM-style loop is encoded.

Flow:
  A. Prime memory
  B. Plan
  C. Search + Read per sub-question
  D. Synthesize
  E. Reflect
  F. Write memory (personal/task/tool to ReMe; working to Qdrant)
"""

from __future__ import annotations

import time
from datetime import datetime

import structlog

from deepresearch.agents.context import RunContext, RunDependencies
from deepresearch.agents.planner import PlannerAgent
from deepresearch.agents.reader import ReaderAgent
from deepresearch.agents.reflector import ReflectorAgent
from deepresearch.agents.searcher import SearcherAgent
from deepresearch.agents.synthesizer import SynthesizerAgent
from deepresearch.config.schema import MemoryProfileConfig
from deepresearch.memory.profiles import MemoryProfile
from deepresearch.schemas.memory import MemoryEventKind, MemoryType
from deepresearch.schemas.runs import ResearchRun, RunRequest, RunStatus

log = structlog.get_logger(__name__)


def _resolve_memory_profile(deps: RunDependencies, name: str) -> MemoryProfile:
    profiles = deps.config.memory.profiles
    if name not in profiles:
        # Fall back to default-equivalent if requested profile is missing.
        return MemoryProfile.from_config(name, MemoryProfileConfig())
    return MemoryProfile.from_config(name, profiles[name])


async def run_research(
    req: RunRequest,
    deps: RunDependencies,
    *,
    existing_run: ResearchRun | None = None,
) -> ResearchRun:
    if existing_run is None:
        run = ResearchRun(request=req, status=RunStatus.pending)
        deps.repos.runs.create(run)
    else:
        run = existing_run
    deps.repos.runs.mark_running(run.id)
    run.started_at = datetime.utcnow()
    run.status = RunStatus.running

    mem_profile = _resolve_memory_profile(deps, req.memory_profile)
    ctx = RunContext(run=run, request=req, deps=deps, memory_profile=mem_profile)
    t_total = time.perf_counter()

    try:
        # A. Prime memory
        ctx.primes = await deps.memory.query_prime(
            run_id=run.id,
            user_id=req.user_id,
            project_id=req.project_id,
            query=req.query,
            profile=mem_profile,
        )
        ctx.run.metrics.n_memory_reads = sum(len(v) for v in ctx.primes.values())

        # B. Plan
        await PlannerAgent().run(ctx, seq=ctx.next_seq())
        if not ctx.plan:
            raise RuntimeError("planner produced no sub-questions")

        # C. Search + Read (cap by max_searches)
        limit = min(len(ctx.plan), req.max_searches)
        for sq in ctx.plan[:limit]:
            await SearcherAgent(sq).run(ctx, seq=ctx.next_seq())
            await ReaderAgent(sq).run(ctx, seq=ctx.next_seq())

        # D. Synthesize
        await SynthesizerAgent().run(ctx, seq=ctx.next_seq())

        # E. Reflect
        await ReflectorAgent().run(ctx, seq=ctx.next_seq())

        # F. Write memories from reflection + persist working trace
        if ctx.reflection:
            for mt, text in (
                (MemoryType.personal, ctx.reflection.personal_update),
                (MemoryType.task, ctx.reflection.task_update),
                (MemoryType.tool, ctx.reflection.tool_update),
            ):
                if text:
                    await deps.memory.write(
                        run_id=run.id,
                        user_id=req.user_id,
                        project_id=req.project_id,
                        memory_type=mt,
                        content=text,
                        metadata={"run_id": str(run.id), "query": req.query},
                    )
                    ctx.run.metrics.n_memory_writes += 1
        # Working memory: store the report + plan summary
        if ctx.draft_report:
            await deps.memory.write(
                run_id=run.id,
                user_id=req.user_id,
                project_id=req.project_id,
                memory_type=MemoryType.working,
                content=ctx.draft_report,
                metadata={"run_id": str(run.id), "query": req.query, "kind": "report"},
                kind=MemoryEventKind.working_write,
            )
            ctx.run.metrics.n_memory_writes += 1

        ctx.run.report_md = ctx.draft_report
        ctx.run.citations = ctx.citations
        ctx.run.status = RunStatus.done
    except Exception as e:
        log.exception("run_failed", run_id=str(run.id), error=repr(e))
        ctx.run.status = RunStatus.failed
        ctx.run.error = repr(e)
    finally:
        ctx.run.finished_at = datetime.utcnow()
        ctx.run.metrics.total_latency_ms = int((time.perf_counter() - t_total) * 1000)
        deps.repos.runs.mark_done(ctx.run)
    return ctx.run
