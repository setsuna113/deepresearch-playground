"""Memory hooks bridging `MemoryService` to the LangGraph pipeline.

Three responsibilities:

1. **`prime_brief_messages`** — pre-graph: pull personal/task/tool/working
   memories via `MemoryService.query_prime` and render them as a system
   message that the upstream `clarify_with_user` /
   `write_research_brief` will see in `state["messages"]`. The first
   element of the returned list is prepended to whatever the caller
   passes as the user query.

2. **`write_reflection`** — post-graph: take a `ReflectionUpdate` and
   write its `personal_update` / `task_update` / `tool_update` strings
   to ReMe via `MemoryService.write`. Counts increments are stamped on
   `RunMetrics` by the caller.

3. **`write_working_report`** — post-graph: persist the synthesized
   report to working memory (our Qdrant) so a future run priming with
   `working_top_k > 0` can retrieve it as part of `query_prime`.

These are deliberately tiny adapters; the real logic lives in
`MemoryService` and `ReMeAdapter`.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import structlog
from langchain_core.messages import SystemMessage

from deepresearch.agents.context import RunDependencies
from deepresearch.memory.profiles import MemoryProfile
from deepresearch.schemas.agents import ReflectionUpdate
from deepresearch.schemas.memory import MemoryEventKind, MemoryRecord, MemoryType
from deepresearch.schemas.runs import RunRequest

log = structlog.get_logger(__name__)


_PRIME_HEADER = (
    "You have access to prior research memory. Use the bullets below if "
    "they appear relevant to the new query; otherwise ignore them.\n"
)


def _render_bullets(records: list[MemoryRecord], label: str) -> str | None:
    if not records:
        return None
    bullets = "\n".join(f"- {(r.content or '').strip()}" for r in records if r.content)
    if not bullets:
        return None
    return f"\n## {label}\n{bullets}"


async def prime_brief_messages(
    *,
    deps: RunDependencies,
    request: RunRequest,
    run_id: UUID,
    profile: MemoryProfile,
) -> tuple[list[SystemMessage], dict[MemoryType, list[MemoryRecord]]]:
    """Return (system_messages, primes_dict).

    `system_messages` is empty if no memory layer is active or the profile
    asks for zero reads of every type. Otherwise it contains one
    `SystemMessage` summarizing the primed memories, suitable for
    prepending to LangGraph's initial state.
    """
    primes = await deps.memory.query_prime(
        run_id=run_id,
        user_id=request.user_id,
        project_id=request.project_id,
        query=request.query,
        profile=profile,
    )
    if not any(primes.values()):
        return [], primes

    sections: list[str] = [_PRIME_HEADER]
    for mt, label in (
        (MemoryType.personal, "Personal preferences and constraints"),
        (MemoryType.task, "Lessons from prior similar research tasks"),
        (MemoryType.tool, "Tool-usage hints"),
        (MemoryType.working, "Prior reports from related runs"),
    ):
        rendered = _render_bullets(primes.get(mt, []), label)
        if rendered:
            sections.append(rendered)
    if len(sections) == 1:  # only header, no actual bullets
        return [], primes
    body = "".join(sections)
    return [SystemMessage(content=body)], primes


async def write_reflection(
    *,
    deps: RunDependencies,
    run_id: UUID,
    request: RunRequest,
    reflection: ReflectionUpdate,
    metadata_extra: dict[str, Any] | None = None,
) -> int:
    """Write each non-empty reflection slot to ReMe. Returns count written."""
    n_written = 0
    base_meta = {"run_id": str(run_id), "query": request.query}
    if metadata_extra:
        base_meta = {**base_meta, **metadata_extra}

    for mt, text in (
        (MemoryType.personal, reflection.personal_update),
        (MemoryType.task, reflection.task_update),
        (MemoryType.tool, reflection.tool_update),
    ):
        if not text:
            continue
        rec = await deps.memory.write(
            run_id=run_id,
            user_id=request.user_id,
            project_id=request.project_id,
            memory_type=mt,
            content=text,
            metadata=base_meta,
        )
        if rec is not None:
            n_written += 1
    return n_written


async def write_working_report(
    *,
    deps: RunDependencies,
    run_id: UUID,
    request: RunRequest,
    report: str,
    extra_metadata: dict[str, Any] | None = None,
) -> MemoryRecord | None:
    """Persist the final report to working memory (our Qdrant)."""
    if not report:
        return None
    meta = {"run_id": str(run_id), "query": request.query, "kind": "report"}
    if extra_metadata:
        meta = {**meta, **extra_metadata}
    return await deps.memory.write(
        run_id=run_id,
        user_id=request.user_id,
        project_id=request.project_id,
        memory_type=MemoryType.working,
        content=report,
        metadata=meta,
        kind=MemoryEventKind.working_write,
    )


__all__ = [
    "prime_brief_messages",
    "write_reflection",
    "write_working_report",
]
