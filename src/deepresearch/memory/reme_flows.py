"""Flow-name constants and payload helpers for `ReMeApp.async_execute`.

ReMe ships a registry of named flows (`reme_ai/config/default.yaml`).
Each flow expects a specific kwarg shape; we centralize the mapping
here so `reme_adapter.py` stays compact and the brittle bits stay in
one place.

Retrieval flows return memories in `result["metadata"]["memory_list"]`
as a list of `BaseMemory`-typed dicts:

    {
        "memory_id": "<uuid>",
        "when_to_use": "<description>",
        "content": "<actual content>",
        "score": <float>,
        ...
    }

Summary flows write to the underlying vector store; their result
metadata does not consistently surface the inserted memory IDs (as of
reme-ai 0.3.x). The adapter therefore synthesizes a fresh
`MemoryRecord` from the input on success.
"""

from __future__ import annotations

from deepresearch.schemas.memory import MemoryType

# ---- Retrieval (read) flows -------------------------------------------------

RETRIEVE_FLOW_FOR: dict[MemoryType, str] = {
    MemoryType.personal: "retrieve_personal_memory",
    MemoryType.task: "retrieve_task_memory",
    MemoryType.tool: "retrieve_tool_memory",
}


# ---- Summary (write) flows --------------------------------------------------
#
# tool memory writes are intentionally absent — ReMe's tool flows take
# `tool_call_results` (a structured list of tool invocations with
# inputs/outputs/timings), not the free-text reflection our pipeline
# emits. Until we wire up an actual tool-call recorder, tool writes
# are no-ops (logged at warning level).
SUMMARY_FLOW_FOR: dict[MemoryType, str | None] = {
    MemoryType.personal: "summary_personal_memory",
    MemoryType.task: "summary_task_memory_simple",
    MemoryType.tool: None,
}


def build_retrieve_kwargs(
    *,
    memory_type: MemoryType,
    workspace_id: str,
    query: str,
    top_k: int,
    tool_names: str | None = None,
) -> dict | None:
    """Build the kwargs payload for a retrieve flow.

    Returns None if the request can't be satisfied (e.g. tool retrieval
    with no `tool_names`).
    """
    if memory_type == MemoryType.tool:
        if not tool_names:
            return None
        return {"workspace_id": workspace_id, "tool_names": tool_names, "top_k": top_k}
    return {"workspace_id": workspace_id, "query": query, "top_k": top_k}


def build_summary_trajectory(*, content: str, score: float = 1.0) -> list[dict]:
    """Wrap a single text blob as a trajectory the summary flows accept.

    ReMe's summary flows expect `trajectories`, a list of
    conversation traces. We have one reflection sentence per slot, so
    we synthesize a minimal trace.
    """
    return [
        {
            "messages": [
                {"role": "user", "content": content},
            ],
            "score": score,
        }
    ]


__all__ = [
    "RETRIEVE_FLOW_FOR",
    "SUMMARY_FLOW_FOR",
    "build_retrieve_kwargs",
    "build_summary_trajectory",
]
