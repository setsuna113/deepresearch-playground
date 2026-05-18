"""Runtime entry: build and run the LangGraph deep-research pipeline.

`run_research(req, deps)` is what `agents.orchestrator.run_research`
(the shim) delegates to. The function:

1. Allocates / loads the `ResearchRun`.
2. Resolves the memory profile and primes personal/task/tool/working
   memory via `MemoryService.query_prime`.
3. Builds the LangGraph (upstream nodes + our injected reflector).
4. Enters the `active_run_context` so `RouterChatModel` resolves to a
   per-run `RouterConfigurableModel`.
5. Invokes the graph with a `RunnableConfig` that
   (a) carries our `TraceCallbackHandler` for SQLite trace recording,
   (b) overrides upstream's `Configuration` fields to use OUR role
       strings: `research_model="supervisor"`,
       `summarization_model="compressor"`, etc.
6. Reads `state["final_report"]` and `state["reflection"]`, persists
   the report and reflection via `MemoryService.write`, finalizes the
   `ResearchRun`.

Anything that previously emitted `AgentStep` rows (the 5 STORM agents)
is now produced by `TraceCallbackHandler` instead.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import structlog
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from deepresearch.agents.context import RunContext, RunDependencies
from deepresearch.agents.langgraph.callbacks import (
    SeqAllocator,
    TraceCallbackHandler,
)
from deepresearch.agents.langgraph.memory_hooks import (
    prime_brief_messages,
    write_reflection,
    write_working_report,
)
from deepresearch.agents.langgraph.reflection_node import reflector_node
from deepresearch.agents.langgraph.role_map import CONFIG_FIELD_TO_ROLE
from deepresearch.agents.langgraph.router_chat_model import active_run_context
from deepresearch.agents.langgraph.state import AgentState
from deepresearch.agents.langgraph.upstream.configuration import Configuration, SearchAPI
from deepresearch.agents.langgraph.upstream.deep_researcher import (
    clarify_with_user,
    final_report_generation,
    supervisor_subgraph,
    write_research_brief,
)
from deepresearch.agents.langgraph.upstream.state import (
    AgentInputState,
)
from deepresearch.config.schema import MemoryProfileConfig
from deepresearch.memory.profiles import MemoryProfile
from deepresearch.schemas.agents import ReflectionUpdate
from deepresearch.schemas.runs import ResearchRun, RunRequest, RunStatus

log = structlog.get_logger(__name__)


def _resolve_memory_profile(deps: RunDependencies, name: str) -> MemoryProfile:
    profiles = deps.config.memory.profiles
    if name not in profiles:
        return MemoryProfile.from_config(name, MemoryProfileConfig())
    return MemoryProfile.from_config(name, profiles[name])


def _build_graph() -> Any:
    """Compile the top-level deep-research graph.

    Re-uses upstream's clarify -> brief -> supervisor -> final_report
    chain (the supervisor subgraph spawns researcher subgraphs that
    handle their own ReAct loops), and appends our reflector node
    between `final_report_generation` and `END`.
    """
    g = StateGraph(AgentState, input_schema=AgentInputState, context_schema=Configuration)
    g.add_node("clarify_with_user", clarify_with_user)
    g.add_node("write_research_brief", write_research_brief)
    g.add_node("research_supervisor", supervisor_subgraph)
    g.add_node("final_report_generation", final_report_generation)
    g.add_node("reflector", reflector_node)

    g.add_edge(START, "clarify_with_user")
    g.add_edge("research_supervisor", "final_report_generation")
    g.add_edge("final_report_generation", "reflector")
    g.add_edge("reflector", END)
    return g.compile()


# Compile once per process. Each invocation gets its own run_id /
# callbacks / contextvar — the compiled graph itself is stateless.
_graph = _build_graph()


def _build_configurable(req: RunRequest) -> dict[str, Any]:
    """Build the `configurable` dict passed to upstream's Configuration.

    Maps upstream model fields onto our role strings (so RouterChatModel
    receives the right `role` via with_config), turns off clarification
    + MCP, and selects no search API for the Phase-1.5 smoke gate (the
    bundled smoke layers on Tavily). Caller can override via
    `req`-derived knobs in future phases.
    """
    return {
        # Model selection — role names that RouterChatModel resolves.
        "research_model": CONFIG_FIELD_TO_ROLE["research_model"],
        "research_model_max_tokens": 1500,
        "summarization_model": CONFIG_FIELD_TO_ROLE["summarization_model"],
        "summarization_model_max_tokens": 800,
        "compression_model": CONFIG_FIELD_TO_ROLE["compression_model"],
        "compression_model_max_tokens": 1200,
        "final_report_model": CONFIG_FIELD_TO_ROLE["final_report_model"],
        # 1500 fits inside Qwen3-8B-AWQ's 4096 context with room for the
        # prompt + findings. For reasoning models (DeepSeek v4-pro etc.)
        # that consume part of the output budget on `reasoning_content`,
        # `ModelClient.complete` falls back to reasoning_content when
        # `content` ends up empty so the report still has substance.
        "final_report_model_max_tokens": 1500,
        # Pipeline behavior.
        "allow_clarification": False,
        "max_concurrent_research_units": req.max_concurrent_units,
        "max_researcher_iterations": min(req.max_searches, 3),
        "max_react_tool_calls": min(req.max_searches, 4),
        "max_structured_output_retries": 2,
        "max_content_length": 8000,
        # Search & MCP off by default for Phase 1.5. The bundled smoke
        # gate enables Tavily explicitly via env.
        "search_api": SearchAPI.NONE.value,
        "mcp_config": None,
    }


async def run_research(
    req: RunRequest,
    deps: RunDependencies,
    *,
    existing_run: ResearchRun | None = None,
) -> ResearchRun:
    run = existing_run if existing_run is not None else ResearchRun(request=req, status=RunStatus.pending)
    if existing_run is None:
        deps.repos.runs.create(run)

    deps.repos.runs.mark_running(run.id)
    run.started_at = datetime.now(UTC)
    run.status = RunStatus.running

    mem_profile = _resolve_memory_profile(deps, req.memory_profile)
    ctx = RunContext(run=run, request=req, deps=deps, memory_profile=mem_profile)
    seq_alloc = SeqAllocator()
    t_total = time.perf_counter()

    try:
        prime_msgs, primes = await prime_brief_messages(
            deps=deps, request=req, run_id=run.id, profile=mem_profile
        )
        ctx.primes = primes
        run.metrics.n_memory_reads = sum(len(v) for v in primes.values())

        initial_messages: list[Any] = list(prime_msgs)
        initial_messages.append(HumanMessage(content=req.query))
        initial_state = {"messages": initial_messages}

        cb = TraceCallbackHandler(
            recorder=deps.recorder,
            run_id=run.id,
            seq_alloc=seq_alloc,
        )
        rconfig: RunnableConfig = {
            "callbacks": [cb],
            "recursion_limit": 60,
            "configurable": _build_configurable(req),
        }

        with active_run_context(deps=deps, request=req, run_id=run.id):
            final_state: dict[str, Any] = await _graph.ainvoke(initial_state, config=rconfig)

        report = (final_state.get("final_report") or "").strip()
        run.report_md = report or None
        # citations: upstream final_report has its own citation format
        # baked into the markdown. We don't (yet) parse them into our
        # Citation schema — leave that for Phase 2 evaluation.

        reflection_dict = final_state.get("reflection") or {}
        reflection = (
            ReflectionUpdate(**reflection_dict)
            if isinstance(reflection_dict, dict)
            else ReflectionUpdate()
        )
        n_writes = await write_reflection(
            deps=deps, run_id=run.id, request=req, reflection=reflection
        )
        run.metrics.n_memory_writes += n_writes

        if report:
            rec = await write_working_report(
                deps=deps, run_id=run.id, request=req, report=report
            )
            if rec is not None:
                run.metrics.n_memory_writes += 1

        run.status = RunStatus.done
    except Exception as e:
        log.exception("run_failed", run_id=str(run.id), error=repr(e))
        run.status = RunStatus.failed
        run.error = repr(e)
    finally:
        run.finished_at = datetime.now(UTC)
        run.metrics.total_latency_ms = int((time.perf_counter() - t_total) * 1000)
        try:
            deps.repos.runs.mark_done(run)
        except Exception as e:  # pragma: no cover - never crash on persist
            log.warning("run_persist_failed", error=repr(e))
    return run


__all__ = ["run_research"]
