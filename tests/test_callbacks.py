"""Unit tests for `TraceCallbackHandler`.

We synthesize LangGraph callback events directly (rather than spinning
up a full graph) to verify the mapping logic in isolation.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from deepresearch.agents.langgraph.callbacks import (
    SeqAllocator,
    TraceCallbackHandler,
)
from deepresearch.schemas.agents import AgentRole, AgentStep, StepStatus


class _FakeRecorder:
    def __init__(self) -> None:
        self.steps: list[AgentStep] = []

    def record_step(self, step: AgentStep) -> None:
        self.steps.append(step)


def _make_handler(run_id: UUID | None = None) -> tuple[TraceCallbackHandler, _FakeRecorder]:
    recorder = _FakeRecorder()
    handler = TraceCallbackHandler(
        recorder=recorder,  # type: ignore[arg-type]
        run_id=run_id or uuid4(),
        seq_alloc=SeqAllocator(),
    )
    return handler, recorder


def _meta(node_name: str) -> dict:
    return {"langgraph_node": node_name}


@pytest.mark.asyncio
async def test_callback_emits_one_step_per_known_node() -> None:
    handler, rec = _make_handler()
    rid = uuid4()
    await handler.on_chain_start(
        serialized=None,
        inputs={"messages": ["hi"]},
        run_id=rid,
        metadata=_meta("supervisor"),
    )
    await handler.on_chain_end(outputs={"messages": ["resp"]}, run_id=rid)

    assert len(rec.steps) == 1
    step = rec.steps[0]
    assert step.role == AgentRole.supervisor
    assert step.status == StepStatus.ok
    assert step.input["parent_seq"] is None
    assert step.seq == 1
    assert step.output == {"messages": ["resp"]}


@pytest.mark.asyncio
async def test_callback_ignores_internal_langgraph_nodes() -> None:
    handler, rec = _make_handler()
    rid = uuid4()
    await handler.on_chain_start(
        serialized=None, inputs={}, run_id=rid, metadata=_meta("__start__")
    )
    await handler.on_chain_end(outputs={}, run_id=rid)
    assert rec.steps == []


@pytest.mark.asyncio
async def test_callback_assigns_parent_seq_for_nested_chains() -> None:
    handler, rec = _make_handler()
    parent_rid, child_rid = uuid4(), uuid4()

    await handler.on_chain_start(
        serialized=None, inputs={}, run_id=parent_rid, metadata=_meta("supervisor")
    )
    await handler.on_chain_start(
        serialized=None,
        inputs={},
        run_id=child_rid,
        parent_run_id=parent_rid,
        metadata=_meta("researcher"),
    )
    await handler.on_chain_end(outputs={"ok": True}, run_id=child_rid, parent_run_id=parent_rid)
    await handler.on_chain_end(outputs={"ok": True}, run_id=parent_rid)

    assert {s.role for s in rec.steps} == {AgentRole.supervisor, AgentRole.researcher}
    by_role = {s.role: s for s in rec.steps}
    assert by_role[AgentRole.researcher].input["parent_seq"] == by_role[AgentRole.supervisor].seq
    # Seqs are monotonic in the order chains START, not end.
    seqs = sorted(s.seq for s in rec.steps)
    assert seqs == [1, 2]


@pytest.mark.asyncio
async def test_callback_handles_parallel_researchers() -> None:
    """Supervisor spawns 3 concurrent researchers, all with parent=supervisor."""
    handler, rec = _make_handler()
    sup_rid = uuid4()
    await handler.on_chain_start(
        serialized=None, inputs={}, run_id=sup_rid, metadata=_meta("supervisor")
    )

    researcher_rids = [uuid4(), uuid4(), uuid4()]
    for rid in researcher_rids:
        await handler.on_chain_start(
            serialized=None,
            inputs={"topic": "X"},
            run_id=rid,
            parent_run_id=sup_rid,
            metadata=_meta("researcher"),
        )
    # Out-of-order completions:
    for rid in reversed(researcher_rids):
        await handler.on_chain_end(outputs={"notes": "..."}, run_id=rid, parent_run_id=sup_rid)
    await handler.on_chain_end(outputs={}, run_id=sup_rid)

    roles = [s.role for s in rec.steps]
    assert roles.count(AgentRole.researcher) == 3
    assert roles.count(AgentRole.supervisor) == 1

    # All researchers share the supervisor's seq as parent_seq.
    supervisor_seq = next(s.seq for s in rec.steps if s.role == AgentRole.supervisor)
    for s in rec.steps:
        if s.role == AgentRole.researcher:
            assert s.input["parent_seq"] == supervisor_seq

    # Seqs are unique across all 4 steps.
    all_seqs = [s.seq for s in rec.steps]
    assert len(all_seqs) == len(set(all_seqs))


@pytest.mark.asyncio
async def test_callback_records_error_status_on_chain_error() -> None:
    handler, rec = _make_handler()
    rid = uuid4()
    await handler.on_chain_start(
        serialized=None, inputs={}, run_id=rid, metadata=_meta("final_report_generation")
    )
    await handler.on_chain_error(error=RuntimeError("boom"), run_id=rid)

    assert len(rec.steps) == 1
    s = rec.steps[0]
    assert s.status == StepStatus.error
    assert "RuntimeError" in s.error  # type: ignore[operator]


@pytest.mark.asyncio
async def test_callback_links_model_call_ids_to_active_frame() -> None:
    handler, rec = _make_handler()
    rid = uuid4()
    await handler.on_chain_start(
        serialized=None, inputs={}, run_id=rid, metadata=_meta("supervisor")
    )

    call_uuid = uuid4()
    gen = ChatGeneration(
        message=AIMessage(content="ok"),
        generation_info={"call_id": str(call_uuid)},
    )
    await handler.on_llm_end(
        response=LLMResult(generations=[[gen]]),
        run_id=uuid4(),  # any child id
        parent_run_id=rid,
    )
    await handler.on_chain_end(outputs={}, run_id=rid)

    assert len(rec.steps) == 1
    assert rec.steps[0].model_call_ids == [call_uuid]


@pytest.mark.asyncio
async def test_callback_records_tool_calls_inside_frame() -> None:
    handler, rec = _make_handler()
    rid = uuid4()
    await handler.on_chain_start(
        serialized=None, inputs={}, run_id=rid, metadata=_meta("researcher")
    )
    await handler.on_tool_start(
        serialized={"name": "web_search"},
        input_str="latency 4090",
        run_id=uuid4(),
        parent_run_id=rid,
    )
    await handler.on_tool_end(output="results...", run_id=uuid4(), parent_run_id=rid)
    await handler.on_chain_end(outputs={"notes": "x"}, run_id=rid)

    s = rec.steps[0]
    assert "tool_calls" in s.output
    assert s.output["tool_calls"][0]["name"] == "web_search"
    assert s.output["tool_calls"][0]["status"] == "ok"


def test_seq_allocator_is_monotonic() -> None:
    alloc = SeqAllocator()
    assert alloc.next() == 1
    assert alloc.next() == 2
    assert alloc.peek() == 2
