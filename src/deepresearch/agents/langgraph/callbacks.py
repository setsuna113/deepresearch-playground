"""`TraceCallbackHandler` — bridges LangGraph events to our SQLite trace.

LangChain fires `on_chain_start` / `on_chain_end` for every node the
graph executes, with a unique LangChain run_id per node invocation. We
key in-flight frames on that id (a dict, not a stack) so concurrent
researcher subgraphs don't clobber each other.

Each frame accumulates the LangChain run ids of its child LLM calls
(`on_chat_model_end` writes the `call_id` from `generation_info` into
the active frame's `model_call_ids`). When the chain ends, we emit a
single `AgentStep` row via `TraceRecorder.record_step`.

Roles come from `role_map.NODE_TO_ROLE` — internal LangGraph nodes (e.g.
`__start__`, `LangGraph`) and unknown sub-chains are silently ignored.

The handler is thread-safe enough for asyncio: all bookkeeping is
single-threaded mutations of dict/list state under the event loop.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import structlog
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from deepresearch.agents.langgraph.role_map import NODE_TO_ROLE, role_for_node
from deepresearch.observability.trace import TraceRecorder
from deepresearch.schemas.agents import AgentRole, AgentStep, StepStatus

log = structlog.get_logger(__name__)

_MAX_PAYLOAD_CHARS = 4_000


def _truncate(value: Any, max_chars: int = _MAX_PAYLOAD_CHARS) -> Any:
    """Make node inputs/outputs safe for SQLite JSON columns.

    Truncates strings, recursively shrinks dicts/lists. Keeps the shape
    so debugging downstream is straightforward, just bounded in size.
    """
    if isinstance(value, str):
        if len(value) <= max_chars:
            return value
        return value[:max_chars] + f"...<truncated {len(value) - max_chars} chars>"
    if isinstance(value, dict):
        return {k: _truncate(v, max_chars) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_truncate(v, max_chars) for v in value][:50]
    # Pydantic-ish objects, messages, etc. — render via repr to avoid
    # unserializable recursion.
    try:
        from langchain_core.messages import BaseMessage

        if isinstance(value, BaseMessage):
            return {
                "type": value.__class__.__name__,
                "content": _truncate(value.content, max_chars),
            }
    except ImportError:  # pragma: no cover - langchain is a hard dep
        pass
    s = repr(value)
    if len(s) > max_chars:
        s = s[:max_chars] + "...<truncated>"
    return s


@dataclass
class _Frame:
    """In-flight state for a single LangGraph node invocation."""

    node_name: str
    role: AgentRole
    seq: int
    parent_seq: int | None
    started_at: datetime
    t0_perf: float
    inputs: dict[str, Any] = field(default_factory=dict)
    model_call_ids: list[UUID] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class TraceCallbackHandler(AsyncCallbackHandler):
    """LangGraph events -> `AgentStep` rows.

    One instance per research run. Holds:

    - `recorder`: where steps are written.
    - `run_id`: our `ResearchRun.id`, attached to every emitted step.
    - `seq_alloc`: thread-safe counter shared with the runtime (so the
      memory-hook / reflector steps share the same monotonic sequence).
    - `frames`: lc_run_id -> in-flight frame.
    - `lc_to_seq`: lc_run_id -> our seq, used to resolve `parent_seq`.
    """

    def __init__(
        self,
        *,
        recorder: TraceRecorder | None,
        run_id: UUID,
        seq_alloc: "SeqAllocator",
    ) -> None:
        super().__init__()
        self._recorder = recorder
        self._run_id = run_id
        self._seq = seq_alloc
        self._frames: dict[UUID, _Frame] = {}
        self._lc_to_seq: dict[UUID, int] = {}
        self._lock = threading.Lock()  # cheap; we are async-single-threaded

    # ---------- chain (node) events ----------
    async def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = self._node_name(serialized, kwargs, metadata)
        role = role_for_node(name) if name else None
        if role is None:
            return  # internal node (e.g., __start__, LangGraph dispatcher)
        seq = self._seq.next()
        parent_seq = self._lc_to_seq.get(parent_run_id) if parent_run_id else None
        with self._lock:
            self._frames[run_id] = _Frame(
                node_name=name,
                role=role,
                seq=seq,
                parent_seq=parent_seq,
                started_at=datetime.now(UTC),
                t0_perf=time.perf_counter(),
                inputs=_truncate(inputs) if isinstance(inputs, dict) else {"_": _truncate(inputs)},
            )
            self._lc_to_seq[run_id] = seq

    async def on_chain_end(
        self,
        outputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        frame = self._frames.pop(run_id, None)
        if frame is None:
            return
        latency_ms = int((time.perf_counter() - frame.t0_perf) * 1000)
        step = AgentStep(
            run_id=self._run_id,
            seq=frame.seq,
            role=frame.role,
            status=StepStatus.ok,
            input={**frame.inputs, "parent_seq": frame.parent_seq},
            output=_wrap_output(outputs, frame.tool_calls),
            started_at=frame.started_at,
            finished_at=datetime.now(UTC),
            latency_ms=latency_ms,
            model_call_ids=list(frame.model_call_ids),
        )
        self._emit(step)

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        frame = self._frames.pop(run_id, None)
        if frame is None:
            return
        latency_ms = int((time.perf_counter() - frame.t0_perf) * 1000)
        step = AgentStep(
            run_id=self._run_id,
            seq=frame.seq,
            role=frame.role,
            status=StepStatus.error,
            input={**frame.inputs, "parent_seq": frame.parent_seq},
            output=_wrap_output(None, frame.tool_calls),
            started_at=frame.started_at,
            finished_at=datetime.now(UTC),
            latency_ms=latency_ms,
            model_call_ids=list(frame.model_call_ids),
            error=repr(error),
        )
        self._emit(step)

    # ---------- LLM events ----------
    async def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        # We don't allocate a step per chat-model invocation; the
        # ModelClient already writes a ModelCallRecord. We just need to
        # link the resulting call_id back to the active frame, which we
        # do in on_llm_end.
        return None

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        frame = self._frame_for_parent(parent_run_id)
        if frame is None:
            return
        for generations in response.generations:
            for gen in generations:
                info = getattr(gen, "generation_info", None) or {}
                cid = info.get("call_id")
                if cid:
                    try:
                        frame.model_call_ids.append(UUID(cid))
                    except (ValueError, TypeError):
                        pass

    # ---------- tool events ----------
    async def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        frame = self._frame_for_parent(parent_run_id)
        if frame is None:
            return
        name = (serialized or {}).get("name") or "tool"
        frame.tool_calls.append({"name": name, "input": _truncate(input_str), "status": "running"})

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        frame = self._frame_for_parent(parent_run_id)
        if frame is None or not frame.tool_calls:
            return
        last = frame.tool_calls[-1]
        last["output"] = _truncate(output)
        last["status"] = "ok"

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        frame = self._frame_for_parent(parent_run_id)
        if frame is None or not frame.tool_calls:
            return
        last = frame.tool_calls[-1]
        last["error"] = repr(error)
        last["status"] = "error"

    # ---------- helpers ----------
    def _emit(self, step: AgentStep) -> None:
        if self._recorder is None:
            return
        try:
            self._recorder.record_step(step)
        except Exception as e:  # pragma: no cover - never fail user run
            log.warning("trace_emit_failed", error=repr(e), seq=step.seq, role=step.role.value)

    def _frame_for_parent(self, parent_run_id: UUID | None) -> _Frame | None:
        """Walk up from a child callback to find the nearest tracked frame.

        LLM/tool callbacks fire under their parent chain's run_id; if
        that's a frame we track, great. Otherwise we keep walking via
        `_lc_to_seq` chains. For now a single-hop lookup covers all
        observed cases.
        """
        if parent_run_id is None:
            return None
        return self._frames.get(parent_run_id)

    @staticmethod
    def _node_name(
        serialized: dict[str, Any] | None,
        kwargs: dict[str, Any],
        metadata: dict[str, Any] | None,
    ) -> str | None:
        # LangGraph's `add_node(name, fn)` surfaces in three places:
        # serialized["name"], kwargs["name"], or metadata["langgraph_node"].
        # Try each in turn.
        name = (
            (serialized or {}).get("name")
            or kwargs.get("name")
            or (metadata or {}).get("langgraph_node")
        )
        if isinstance(name, str):
            return name
        return None


def _wrap_output(outputs: Any, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if outputs is not None:
        if isinstance(outputs, dict):
            payload.update(_truncate(outputs))
        else:
            payload["_output"] = _truncate(outputs)
    if tool_calls:
        payload["tool_calls"] = tool_calls
    return payload


class SeqAllocator:
    """Monotonic seq allocator shared between callback handler and the
    runtime's memory hooks. asyncio-safe single-thread mutex.
    """

    def __init__(self, start: int = 0) -> None:
        self._n = start
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            self._n += 1
            return self._n

    def peek(self) -> int:
        return self._n


__all__ = ["TraceCallbackHandler", "SeqAllocator", "NODE_TO_ROLE"]
