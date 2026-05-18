# ruff: noqa: E402 — the `os.environ.setdefault(...)` block intentionally
# precedes the langchain/langgraph imports so any env-driven defaults in
# upstream `Configuration.from_runnable_config` are in place by the time
# the graph is compiled at module load.
"""LangGraph Studio entry point.

Upstream `open_deep_research` exposes its compiled `deep_researcher`
graph via a `langgraph.json` so `langgraph dev` can serve it in the
Studio web UI. Our `runtime.run_research` wraps the same graph with
contextvar setup + memory priming + reflection writes, which makes it
unsuitable for Studio's "just hand me a compiled graph" API.

This module exposes a `studio_graph` that mirrors `runtime.run_research`
for the Studio path:

1. `studio_bootstrap` lazy-builds `RunDependencies` once per process,
   allocates a `ResearchRun`, primes personal/task/tool/working memory
   into a system message, builds a `TraceCallbackHandler` + `SeqAllocator`,
   binds the module-level Studio active-run slot, and tries to attach
   the callback into the local `RunnableConfig` so Pregel propagates it
   to siblings.
2. Standard upstream nodes (clarify, brief, supervisor) run; every LLM
   call resolves to our `RouterChatModel` via the active-run slot.
3. The new `reflector_writer_node` runs `reflector_node`'s reflection,
   then writes both the reflection and the final report through
   `MemoryService.write` — Studio sessions become first-class for
   thesis data collection.

Run with:

    cp config/config.example.yaml config/config.local.yaml
    # edit local.yaml so models.endpoints.local.base_url points at a
    # real OpenAI-compatible endpoint (vLLM, hosted, etc.)
    uvx --refresh --from "langgraph-cli[inmem]" --with-editable . \\
        --python 3.12 langgraph dev --allow-blocking

LangGraph Studio opens in your browser. Pick "Deep Researcher (Phase 1.5)"
and submit a query.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import UTC, datetime
from typing import Any

import structlog

# Override upstream `Configuration` defaults at import time. The
# upstream `from_runnable_config` reads `os.environ.get(FIELD.upper(),
# configurable.get(field))` so env vars win over the upstream defaults
# of "openai:gpt-4.1". Our RouterChatModel needs role names, not real
# model strings — these env vars wire that up.
#
# `os.environ.setdefault` means users can still override per-process
# via a `.env` file or shell exports (the langgraph dev server loads
# `.env` automatically).
for k, v in (
    ("RESEARCH_MODEL", "supervisor"),
    ("SUMMARIZATION_MODEL", "compressor"),
    ("COMPRESSION_MODEL", "compressor"),
    ("FINAL_REPORT_MODEL", "final_report"),
    ("ALLOW_CLARIFICATION", "false"),
    ("MAX_CONCURRENT_RESEARCH_UNITS", "2"),
    ("MAX_RESEARCHER_ITERATIONS", "3"),
    ("MAX_REACT_TOOL_CALLS", "4"),
    ("MAX_STRUCTURED_OUTPUT_RETRIES", "2"),
    ("MAX_CONTENT_LENGTH", "8000"),
    ("SEARCH_API", "none"),
    # Cap per-call output budgets so requests fit inside the local
    # 8B-AWQ model's 4096-token context. Upstream defaults are 10000,
    # which vLLM rejects with HTTP 400. ModelClient falls back to
    # `reasoning_content` when reasoning models burn the budget.
    ("RESEARCH_MODEL_MAX_TOKENS", "1500"),
    ("SUMMARIZATION_MODEL_MAX_TOKENS", "800"),
    ("COMPRESSION_MODEL_MAX_TOKENS", "1200"),
    ("FINAL_REPORT_MODEL_MAX_TOKENS", "1500"),
):
    os.environ.setdefault(k, v)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from deepresearch.agents.context import RunDependencies
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
from deepresearch.agents.langgraph.router_chat_model import (
    _ActiveRun,
    set_studio_active_run,
)
from deepresearch.agents.langgraph.state import AgentState
from deepresearch.agents.langgraph.upstream.configuration import (
    Configuration,
)
from deepresearch.agents.langgraph.upstream.deep_researcher import (
    clarify_with_user,
    final_report_generation,
    supervisor_subgraph,
    write_research_brief,
)
from deepresearch.agents.langgraph.upstream.state import (
    AgentInputState,
)
from deepresearch.api.deps import build_dependencies
from deepresearch.config.loader import get_config
from deepresearch.config.schema import MemoryProfileConfig
from deepresearch.memory.profiles import MemoryProfile
from deepresearch.schemas.agents import AgentRole, AgentStep, ReflectionUpdate, StepStatus
from deepresearch.schemas.runs import ResearchRun, RunRequest, RunStatus

log = structlog.get_logger(__name__)


_DEPS_CACHE: RunDependencies | None = None
_DEPS_LOCK = asyncio.Lock()


async def _get_deps() -> RunDependencies:
    """Build RunDependencies once per process and re-use.

    Studio re-invokes the graph on every user turn; we want to share the
    SQLite engine + Qdrant connection + ModelClient HTTP pool across
    invocations.
    """
    global _DEPS_CACHE
    if _DEPS_CACHE is not None:
        return _DEPS_CACHE
    async with _DEPS_LOCK:
        if _DEPS_CACHE is None:
            cfg = get_config()
            _DEPS_CACHE = await build_dependencies(cfg)
            log.info("studio_deps_initialized")
    return _DEPS_CACHE


def _query_from_state(state: dict[str, Any]) -> str:
    """Pull the most recent HumanMessage out of state.messages."""
    messages = state.get("messages") or []
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
        # LangGraph Studio may serialize messages as dicts.
        if isinstance(m, dict) and m.get("type") in {"human", "user"}:
            return str(m.get("content", ""))
    return ""


def _resolve_memory_profile(deps: RunDependencies, name: str) -> MemoryProfile:
    profiles = deps.config.memory.profiles
    if name not in profiles:
        return MemoryProfile.from_config(name, MemoryProfileConfig())
    return MemoryProfile.from_config(name, profiles[name])


async def studio_bootstrap(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """First-node setup: deps, run row, memory priming, trace callback.

    Mirrors `runtime.run_research`'s setup for the Studio path. After
    this node returns, every downstream LLM call resolves through the
    `_ActiveRun` slot (which now also carries the trace callback
    handler) and the primed system message is prepended onto messages
    fed to RouterChatModel by the upstream nodes.
    """
    deps = await _get_deps()

    query = _query_from_state(state) or "(empty query)"
    cfg_data: dict[str, Any] = {}
    if isinstance(config, dict):
        cfg_data = config.get("configurable", {}) or {}
    req = RunRequest(
        query=query,
        user_id=str(cfg_data.get("user_id") or "studio"),
        project_id=str(cfg_data.get("project_id") or "studio"),
        model_profile=str(cfg_data.get("model_profile") or "phase1_default"),
        memory_profile=str(cfg_data.get("memory_profile") or "default"),
        max_searches=int(cfg_data.get("max_searches") or 3),
        max_concurrent_units=int(cfg_data.get("max_concurrent_units") or 2),
    )

    run = ResearchRun(request=req, status=RunStatus.running)
    try:
        deps.repos.runs.create(run)
        deps.repos.runs.mark_running(run.id)
        run.started_at = datetime.now(UTC)
    except Exception:
        # Persist failure is visible in logs (full traceback) — thesis
        # runs depend on the SQLite trace, so loudness matters here.
        log.exception("studio_run_persist_failed")

    # Build the trace machinery up-front so RouterChatModel calls (and
    # the bootstrap step itself) all land on a single seq sequence.
    seq_alloc = SeqAllocator()
    cb = TraceCallbackHandler(
        recorder=deps.recorder,
        run_id=run.id,
        seq_alloc=seq_alloc,
    )

    # Best-effort: prime personal/task/tool/working memory. If the memory
    # layer is mid-init or read fails, log and continue with no primes.
    mem_profile = _resolve_memory_profile(deps, req.memory_profile)
    prime_msgs: list[SystemMessage] = []
    primes: dict[Any, list[Any]] = {}
    prime_error: str | None = None
    try:
        prime_msgs, primes = await prime_brief_messages(
            deps=deps, request=req, run_id=run.id, profile=mem_profile
        )
    except Exception as e:
        log.exception("studio_prime_failed")
        prime_error = type(e).__name__

    set_studio_active_run(
        _ActiveRun(deps=deps, request=req, run_id=run.id)
    )
    # Stash callback + primes alongside the active run so the materialized
    # RouterChatModel can pick them up. `_ActiveRun` is a tiny dataclass;
    # we add the runtime-only attributes here without expanding its
    # schema so non-Studio callers stay unaffected.
    from deepresearch.agents.langgraph import router_chat_model as _rcm

    active = _rcm._studio_active_run
    if active is not None:
        active.callback_handler = cb  # type: ignore[attr-defined]
        active.seq_alloc = seq_alloc  # type: ignore[attr-defined]
        active.primes = primes  # type: ignore[attr-defined]
        active.prime_msgs = list(prime_msgs)  # type: ignore[attr-defined]
        active.memory_profile = mem_profile  # type: ignore[attr-defined]
        active.t0 = time.perf_counter()  # type: ignore[attr-defined]

    # Attach the callback into the live RunnableConfig so downstream
    # nodes' LangChain callbacks fire chain_start/chain_end into our
    # recorder. Pregel propagates the parent config to child nodes;
    # mutating the callbacks list reaches subsequent nodes in this
    # invocation.
    if isinstance(config, dict):
        cbs = config.setdefault("callbacks", [])
        if isinstance(cbs, list) and cb not in cbs:
            cbs.append(cb)

    # Record a bootstrap step so the SQLite trace has a clear "studio
    # run started" marker even if downstream chain callbacks don't fire.
    # If priming failed, emit the step with status=error and the exception
    # class name in output so thesis traces never silently miss a failed
    # prime read.
    try:
        deps.recorder.record_step(
            AgentStep(
                run_id=run.id,
                seq=seq_alloc.next(),
                role=AgentRole.planner,
                status=StepStatus.error if prime_error else StepStatus.ok,
                input={"node": "studio_bootstrap", "query": query},
                output={
                    "n_primes": sum(len(v) for v in primes.values()),
                    "model_profile": req.model_profile,
                    "memory_profile": req.memory_profile,
                    **({"prime_error": prime_error} if prime_error else {}),
                },
                started_at=run.started_at or datetime.now(UTC),
                finished_at=datetime.now(UTC),
                latency_ms=0,
                error=prime_error,
            )
        )
    except Exception:
        # The recorder itself failed — can't emit a step about a step. Log
        # loudly so the silent-failure surface is at least visible in the
        # process logs.
        log.exception("studio_bootstrap_trace_failed")

    # Studio's add_messages reducer appends rather than prepends, so we
    # can't cleanly inject the prime SystemMessages at position 0 from
    # here. Instead the materialized RouterChatModel reads
    # `_ActiveRun.prime_msgs` and prepends the primed context onto every
    # LLM call's messages. Costs a few hundred prompt tokens per call
    # but matches the runtime path's effect at the model layer.
    return {}


async def reflector_writer_node(
    state: AgentState, config: RunnableConfig
) -> dict[str, Any]:
    """Run reflector + write reflection / report into ReMe + working memory.

    Replaces the bare `reflector_node` in the Studio graph: we still
    emit the same `ReflectionUpdate` into state (so the Studio UI can
    inspect it) but additionally persist it via `MemoryService.write`
    the same way `runtime.run_research` does after a CLI run.
    """
    # Step 1: run the standard reflector to populate state["reflection"].
    result = await reflector_node(state, config)

    # Step 2: dispatch the writes. Read the active-run slot to find
    # deps + run_id. If the slot is empty (e.g., reflector triggered
    # outside Studio), no-op the writes.
    from deepresearch.agents.langgraph import router_chat_model as _rcm

    active = _rcm._studio_active_run
    if active is None:
        return result

    deps = active.deps
    req = active.request
    run_id = active.run_id

    reflection_dict = result.get("reflection") or {}
    reflection = (
        ReflectionUpdate(**reflection_dict)
        if isinstance(reflection_dict, dict)
        else ReflectionUpdate()
    )
    final_report = (state.get("final_report") or "").strip()

    write_errors: list[str] = []
    try:
        n_writes = await write_reflection(
            deps=deps, run_id=run_id, request=req, reflection=reflection
        )
    except Exception as e:
        log.exception("studio_write_reflection_failed")
        write_errors.append(f"reflection:{type(e).__name__}")
        n_writes = 0

    if final_report:
        try:
            await write_working_report(
                deps=deps, run_id=run_id, request=req, report=final_report
            )
            n_writes += 1
        except Exception as e:
            log.exception("studio_write_report_failed")
            write_errors.append(f"report:{type(e).__name__}")

    # Final marker step + finalize the run row so studio sessions appear
    # in `data/runs.db` as `done` rather than stuck `running`.
    seq_alloc = getattr(active, "seq_alloc", None)
    started_at = getattr(active, "t0", None)
    latency_ms = int((time.perf_counter() - started_at) * 1000) if started_at else 0
    try:
        if seq_alloc is not None:
            deps.recorder.record_step(
                AgentStep(
                    run_id=run_id,
                    seq=seq_alloc.next(),
                    role=AgentRole.reflector,
                    status=StepStatus.error if write_errors else StepStatus.ok,
                    input={"node": "reflector_writer"},
                    output={
                        "n_writes": n_writes,
                        "report_chars": len(final_report),
                        **({"write_errors": write_errors} if write_errors else {}),
                    },
                    started_at=datetime.now(UTC),
                    finished_at=datetime.now(UTC),
                    latency_ms=0,
                    error=";".join(write_errors) if write_errors else None,
                )
            )
        # Mark the run done. ResearchRun.report_md is set so future
        # primed runs can retrieve via working memory.
        try:
            from deepresearch.schemas.runs import RunMetrics

            done_run = ResearchRun(
                id=run_id,
                request=req,
                status=RunStatus.done,
                started_at=datetime.now(UTC),
                finished_at=datetime.now(UTC),
                report_md=final_report or None,
                metrics=RunMetrics(
                    total_latency_ms=latency_ms,
                    n_memory_writes=n_writes,
                ),
            )
            deps.repos.runs.mark_done(done_run)
        except Exception:
            log.exception("studio_run_mark_done_failed")
    except Exception:
        log.exception("studio_finalize_trace_failed")

    return result


def _build_studio_graph() -> Any:
    g = StateGraph(AgentState, input_schema=AgentInputState, context_schema=Configuration)
    g.add_node("studio_bootstrap", studio_bootstrap)
    g.add_node("clarify_with_user", clarify_with_user)
    g.add_node("write_research_brief", write_research_brief)
    g.add_node("research_supervisor", supervisor_subgraph)
    g.add_node("final_report_generation", final_report_generation)
    g.add_node("reflector", reflector_writer_node)

    g.add_edge(START, "studio_bootstrap")
    g.add_edge("studio_bootstrap", "clarify_with_user")
    # clarify_with_user / write_research_brief return Commands that
    # route to the next node by name; no explicit edges needed for them.
    g.add_edge("research_supervisor", "final_report_generation")
    g.add_edge("final_report_generation", "reflector")
    g.add_edge("reflector", END)
    return g.compile()


# Exported for langgraph.json -> studio_graph
studio_graph = _build_studio_graph()


__all__ = ["reflector_writer_node", "studio_bootstrap", "studio_graph"]
