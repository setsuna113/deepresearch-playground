"""Unit tests for `ReMeAdapter`.

We mock `ReMeApp.async_execute` and verify that:

- `query` dispatches to the right flow name with the right payload shape.
- Retrieved memories from `result["metadata"]["memory_list"]` are coerced
  to our `MemoryRecord` schema.
- Tool retrieval without `tool_names` returns `[]` (open question 2).
- `write` dispatches to summary flows for personal/task, no-ops for tool.
- Init failures fall through to a disabled adapter without raising.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from deepresearch.config.schema import (
    ReMeEmbeddingConfig,
    ReMeLLMConfig,
    ReMeSection,
    ReMeVectorStoreConfig,
)
from deepresearch.memory.reme_adapter import ReMeAdapter
from deepresearch.memory.reme_flows import (
    RETRIEVE_FLOW_FOR,
    SUMMARY_FLOW_FOR,
)
from deepresearch.schemas.memory import MemoryType


def _make_section(enabled: bool = True) -> ReMeSection:
    return ReMeSection(
        enabled=enabled,
        working_dir="./data/reme",
        vector_store=ReMeVectorStoreConfig(
            backend="qdrant",
            url="http://localhost:6333",
            collection_prefix="dr_reme_",
        ),
        llm=ReMeLLMConfig(endpoint="local"),
        embedding=ReMeEmbeddingConfig(provider="local", model_id="bge-m3"),
    )


@pytest.fixture
async def disabled_adapter() -> ReMeAdapter:
    return await ReMeAdapter.create(_make_section(enabled=False))


@pytest.fixture
async def adapter_with_mock_reme() -> ReMeAdapter:
    """Adapter with a manually-injected mock `ReMeApp` (skips real init)."""
    section = _make_section(enabled=True)
    adapter = ReMeAdapter(section=section)
    adapter._reme = SimpleNamespace(async_execute=AsyncMock(return_value={}))
    return adapter


@pytest.mark.asyncio
async def test_disabled_adapter_no_ops(disabled_adapter) -> None:
    assert disabled_adapter.available is False
    out = await disabled_adapter.query(
        user_id="u", project_id="p", query="q", memory_type=MemoryType.task, top_k=5
    )
    assert out == []
    rec = await disabled_adapter.write(
        user_id="u",
        project_id="p",
        memory_type=MemoryType.task,
        content="hello",
    )
    assert rec is None


@pytest.mark.asyncio
async def test_query_task_dispatches_correct_flow(adapter_with_mock_reme) -> None:
    adapter_with_mock_reme._reme.async_execute.return_value = {
        "metadata": {
            "memory_list": [
                {
                    "memory_id": "m1",
                    "when_to_use": "When the user asks about quantization.",
                    "content": "AWQ tends to win on instruction-following.",
                    "score": 0.81,
                }
            ]
        }
    }

    records = await adapter_with_mock_reme.query(
        user_id="alice",
        project_id="thesis",
        query="AWQ vs GPTQ",
        memory_type=MemoryType.task,
        top_k=3,
    )

    assert len(records) == 1
    assert records[0].id == "m1"
    assert records[0].memory_type == MemoryType.task
    assert "AWQ" in records[0].content
    assert records[0].score == pytest.approx(0.81)

    mock = adapter_with_mock_reme._reme.async_execute
    mock.assert_awaited_once_with(
        RETRIEVE_FLOW_FOR[MemoryType.task],
        workspace_id="alice/thesis",
        query="AWQ vs GPTQ",
        top_k=3,
    )


@pytest.mark.asyncio
async def test_query_personal_dispatches_personal_flow(adapter_with_mock_reme) -> None:
    adapter_with_mock_reme._reme.async_execute.return_value = {
        "metadata": {"memory_list": []}
    }
    await adapter_with_mock_reme.query(
        user_id="u",
        project_id="p",
        query="?",
        memory_type=MemoryType.personal,
        top_k=2,
    )
    flow = adapter_with_mock_reme._reme.async_execute.call_args.args[0]
    assert flow == RETRIEVE_FLOW_FOR[MemoryType.personal]


@pytest.mark.asyncio
async def test_query_tool_without_tool_names_returns_empty(adapter_with_mock_reme) -> None:
    out = await adapter_with_mock_reme.query(
        user_id="u",
        project_id="p",
        query="?",
        memory_type=MemoryType.tool,
        top_k=5,
    )
    assert out == []
    adapter_with_mock_reme._reme.async_execute.assert_not_called()


@pytest.mark.asyncio
async def test_query_tool_with_tool_names(adapter_with_mock_reme) -> None:
    adapter_with_mock_reme._reme.async_execute.return_value = {
        "metadata": {"memory_list": []}
    }
    await adapter_with_mock_reme.query(
        user_id="u",
        project_id="p",
        query="ignored for tool memory",
        memory_type=MemoryType.tool,
        top_k=5,
        tool_names="tavily,fetch",
    )
    kwargs = adapter_with_mock_reme._reme.async_execute.call_args.kwargs
    assert kwargs["tool_names"] == "tavily,fetch"
    assert "query" not in kwargs


@pytest.mark.asyncio
async def test_query_swallows_reme_errors(adapter_with_mock_reme) -> None:
    adapter_with_mock_reme._reme.async_execute.side_effect = RuntimeError("boom")
    out = await adapter_with_mock_reme.query(
        user_id="u", project_id="p", query="q", memory_type=MemoryType.task, top_k=3
    )
    assert out == []


@pytest.mark.asyncio
async def test_query_skips_with_top_k_zero(adapter_with_mock_reme) -> None:
    out = await adapter_with_mock_reme.query(
        user_id="u", project_id="p", query="q", memory_type=MemoryType.task, top_k=0
    )
    assert out == []
    adapter_with_mock_reme._reme.async_execute.assert_not_called()


@pytest.mark.asyncio
async def test_write_task_dispatches_summary_flow(adapter_with_mock_reme) -> None:
    adapter_with_mock_reme._reme.async_execute.return_value = {
        "metadata": {"memory_ids": ["new-id-1"]}
    }
    rec = await adapter_with_mock_reme.write(
        user_id="u",
        project_id="p",
        memory_type=MemoryType.task,
        content="search shallow before going deep",
    )
    assert rec is not None
    assert rec.id == "new-id-1"
    assert rec.memory_type == MemoryType.task

    flow = adapter_with_mock_reme._reme.async_execute.call_args.args[0]
    kwargs = adapter_with_mock_reme._reme.async_execute.call_args.kwargs
    assert flow == SUMMARY_FLOW_FOR[MemoryType.task]
    assert kwargs["workspace_id"] == "u/p"
    assert isinstance(kwargs["trajectories"], list)
    assert kwargs["trajectories"][0]["messages"][0]["content"] == (
        "search shallow before going deep"
    )


@pytest.mark.asyncio
async def test_write_personal_dispatches_personal_summary(adapter_with_mock_reme) -> None:
    adapter_with_mock_reme._reme.async_execute.return_value = {}
    await adapter_with_mock_reme.write(
        user_id="u",
        project_id="p",
        memory_type=MemoryType.personal,
        content="user prefers primary sources",
    )
    flow = adapter_with_mock_reme._reme.async_execute.call_args.args[0]
    assert flow == SUMMARY_FLOW_FOR[MemoryType.personal]


@pytest.mark.asyncio
async def test_write_tool_is_noop(adapter_with_mock_reme) -> None:
    rec = await adapter_with_mock_reme.write(
        user_id="u",
        project_id="p",
        memory_type=MemoryType.tool,
        content="tool insight",
    )
    assert rec is None
    adapter_with_mock_reme._reme.async_execute.assert_not_called()


@pytest.mark.asyncio
async def test_write_swallows_reme_errors(adapter_with_mock_reme) -> None:
    adapter_with_mock_reme._reme.async_execute.side_effect = RuntimeError("flow blew up")
    rec = await adapter_with_mock_reme.write(
        user_id="u",
        project_id="p",
        memory_type=MemoryType.task,
        content="anything",
    )
    assert rec is None


@pytest.mark.asyncio
async def test_init_falls_through_when_reme_constructor_raises(monkeypatch) -> None:
    """If ReMeApp(*args, **kwargs) raises, adapter stays disabled."""
    import deepresearch.memory.reme_adapter as adapter_mod

    class _BadCtor:
        def __init__(self, *a: Any, **k: Any) -> None:
            raise RuntimeError("no embedding endpoint")

    fake_reme = SimpleNamespace(ReMeApp=_BadCtor)
    monkeypatch.setattr(adapter_mod, "_try_import_reme", lambda: fake_reme)

    adapter = await ReMeAdapter.create(_make_section(enabled=True))
    assert adapter.available is False
