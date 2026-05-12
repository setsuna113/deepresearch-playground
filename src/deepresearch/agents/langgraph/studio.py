"""LangGraph Studio entry point.

Upstream `open_deep_research` exposes its compiled `deep_researcher`
graph via a `langgraph.json` so `langgraph dev` can serve it in the
Studio web UI. Our `runtime.run_research` wraps the same graph with
contextvar setup + memory priming + reflection writes, which makes it
unsuitable for Studio's "just hand me a compiled graph" API.

This module exposes a `studio_graph` that adds a single bootstrap node
at the front of the pipeline:

1. Lazy-init `RunDependencies` once per process (real config, real
   SQLite, real Qdrant embedded, real ModelClient pointing at the
   endpoints you configured in `config/config.local.yaml`).
2. Allocate a `ResearchRun` row.
3. Set the `active_run_context` contextvar — without `reset()`, so the
   value propagates to every downstream node in this asyncio task.

Then the standard upstream nodes run, followed by our `reflector`.

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
):
    os.environ.setdefault(k, v)
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from deepresearch.agents.context import RunDependencies
from deepresearch.agents.langgraph.reflection_node import reflector_node
from deepresearch.agents.langgraph.router_chat_model import (
    _ActiveRun,
    set_studio_active_run,
)
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
    AgentState,
)
from deepresearch.api.deps import build_dependencies
from deepresearch.config.loader import get_config
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


async def studio_bootstrap(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """First-node setup: build deps, allocate run, bind contextvar."""
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
    )

    run = ResearchRun(request=req, status=RunStatus.running)
    try:
        deps.repos.runs.create(run)
        deps.repos.runs.mark_running(run.id)
        run.started_at = datetime.now(UTC)
    except Exception as e:  # pragma: no cover - studio resilience
        log.warning("studio_run_persist_failed", error=repr(e))

    # LangGraph's Pregel runs each node in a fresh asyncio Task, so
    # contextvars set here would NOT propagate to clarify_with_user
    # next. Use the module-level Studio slot instead — single-tenant
    # but adequate for the `langgraph dev` use case.
    set_studio_active_run(_ActiveRun(deps=deps, request=req, run_id=run.id))

    return {}


def _build_studio_graph() -> Any:
    g = StateGraph(AgentState, input=AgentInputState, config_schema=Configuration)
    g.add_node("studio_bootstrap", studio_bootstrap)
    g.add_node("clarify_with_user", clarify_with_user)
    g.add_node("write_research_brief", write_research_brief)
    g.add_node("research_supervisor", supervisor_subgraph)
    g.add_node("final_report_generation", final_report_generation)
    g.add_node("reflector", reflector_node)

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


__all__ = ["studio_bootstrap", "studio_graph"]
