"""Reflector node — our injected critic between `final_report_generation`
and `END`.

Reads the final report from `state["final_report"]` and asks the
configured reflector LLM (via `get_active_router_model()`) to emit a
structured `ReflectionUpdate` (personal / task / tool slots). The
update is written back to memory by `runtime.run_research` after the
graph finishes.

The node mutates state with a `reflection` field. We avoid extending
the upstream `AgentState` TypedDict by stashing the structured dict
under a non-typed key — LangGraph's MessagesState-derived state is
permissive about additional keys.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from deepresearch.agents.langgraph.router_chat_model import get_active_router_model
from deepresearch.schemas.agents import ReflectionUpdate

log = structlog.get_logger(__name__)


_REFLECTOR_SYSTEM = """\
You are the Reflector. The user just received a research report. Your
job is to extract REUSABLE LESSONS that should improve future research
for THIS user on RELATED queries.

Return strict JSON with these keys; any field may be null if there is
nothing useful to record:

{
  "personal_update": "<single sentence describing a stable user preference observed in this run, or null>",
  "task_update": "<single sentence describing a research strategy that worked or failed for this task type, or null>",
  "tool_update": "<single sentence describing a tool-usage lesson (search formulation, fetch reliability, etc.), or null>",
  "needs_revision": false
}

Do not include any prose outside the JSON object. Do not duplicate the
research findings — those are already saved as the run report. Be
specific. If nothing is worth recording for a slot, use null.
"""


_REFLECTOR_HUMAN_TEMPLATE = """\
Query:
{query}

Final report:
{report}

Notes (raw research traces, possibly empty):
{notes}
"""


async def reflector_node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    """Reflect on the run and stash a `ReflectionUpdate` into state.

    Returns a partial-state dict consumed by LangGraph's reducer.
    `runtime.run_research` reads `state["reflection"]` after invocation
    and dispatches the writes via `memory_hooks.write_reflection`.
    """
    report = state.get("final_report") or ""
    notes = state.get("notes") or []
    messages = state.get("messages") or []
    # Best-effort query extraction: the first HumanMessage is the user
    # turn that kicked off the graph.
    query = ""
    for m in messages:
        if isinstance(m, HumanMessage):
            query = m.content if isinstance(m.content, str) else str(m.content)
            break

    if not report.strip():
        log.warning("reflector_skipped", reason="empty_final_report")
        return {"reflection": ReflectionUpdate().model_dump()}

    notes_text = "\n".join(notes[:5]) if notes else "(none)"
    human = _REFLECTOR_HUMAN_TEMPLATE.format(
        query=query or "(unknown)",
        report=report[:6000],
        notes=notes_text[:3000],
    )

    chain = get_active_router_model().with_config(
        {
            "model": "reflector",
            "max_tokens": 600,
            "tags": ["langsmith:nostream"],
        }
    )
    try:
        ai = await chain.ainvoke(
            [
                SystemMessage(content=_REFLECTOR_SYSTEM),
                HumanMessage(content=human),
            ]
        )
        update = _parse_reflection(ai.content if hasattr(ai, "content") else str(ai))
    except Exception as e:  # pragma: no cover - resilience
        log.warning("reflector_failed", error=repr(e))
        update = ReflectionUpdate()

    return {"reflection": update.model_dump()}


def _parse_reflection(raw: str) -> ReflectionUpdate:
    """Tolerantly parse JSON out of a (possibly fenced) reflector response."""
    text = (raw or "").strip()
    if text.startswith("```"):
        # Strip ```json ... ``` fences.
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    # Find the outermost {...} if there's prose around it.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return ReflectionUpdate()
    if not isinstance(data, dict):
        return ReflectionUpdate()
    return ReflectionUpdate(
        personal_update=_str_or_none(data.get("personal_update")),
        task_update=_str_or_none(data.get("task_update")),
        tool_update=_str_or_none(data.get("tool_update")),
        needs_revision=bool(data.get("needs_revision", False)),
    )


def _str_or_none(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s or None
    return str(v).strip() or None


__all__ = ["reflector_node"]
