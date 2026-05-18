"""Live ReMe smoke tests.

Skipped by default. Run with:

    LIVE_REME=1 \\
    REME_LLM_API_KEY=... \\
    REME_LLM_API_BASE=... \\
    REME_EMBEDDING_API_BASE=... \\
    REME_EMBEDDING_API_KEY=... \\
    uv run pytest tests/test_reme_live.py -v

The minimum env required is:

- `LIVE_REME=1`
- one of `REME_LLM_API_KEY` / `DEEPSEEK_API_KEY` / `OPENAI_API_KEY`
- one of `REME_EMBEDDING_API_BASE` / `OPENAI_API_KEY`
  (because the adapter falls back the LLM endpoint to OpenAI defaults
  if no explicit base is configured).

Tests exercise:

1. `personal` memory roundtrip — write via `summary_personal_memory`,
   retrieve via `retrieve_personal_memory`.
2. `task` memory roundtrip — write via `summary_task_memory_simple`,
   retrieve via `retrieve_task_memory`.

Both flows talk to a live ReMe instance which in turn talks to a live
LLM + embedding endpoint, so failures here surface real wire-up
problems (auth, model availability, embedding dim mismatches, etc.).
"""

from __future__ import annotations

import os

import pytest

from deepresearch.config.schema import ReMeSection
from deepresearch.memory.reme_adapter import ReMeAdapter
from deepresearch.schemas.memory import MemoryType

pytestmark = pytest.mark.skipif(
    os.environ.get("LIVE_REME") != "1",
    reason="Set LIVE_REME=1 to run live ReMe smoke tests",
)


def _have_llm_creds() -> bool:
    return any(
        os.environ.get(k)
        for k in ("REME_LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY")
    )


def _have_embedding_endpoint() -> bool:
    return bool(
        os.environ.get("REME_EMBEDDING_API_BASE")
        or os.environ.get("OPENAI_API_KEY")
    )


live_only = pytest.mark.skipif(
    not _have_llm_creds() or not _have_embedding_endpoint(),
    reason=(
        "Live ReMe tests need an LLM endpoint (REME_LLM_API_KEY or "
        "DEEPSEEK_API_KEY or OPENAI_API_KEY) and an embedding endpoint "
        "(REME_EMBEDDING_API_BASE or OPENAI_API_KEY)"
    ),
)


@pytest.fixture
async def adapter() -> ReMeAdapter:
    section = ReMeSection(enabled=True)
    adapter = await ReMeAdapter.create(section)
    if not adapter.available:
        pytest.skip("ReMeAdapter failed to initialize; check creds + endpoints")
    return adapter


@live_only
@pytest.mark.asyncio
async def test_reme_personal_roundtrip(adapter: ReMeAdapter) -> None:
    """Write a personal memory, then retrieve it via a related query."""
    user_id = "live_test_user"
    project_id = "live_test"

    rec = await adapter.write(
        user_id=user_id,
        project_id=project_id,
        memory_type=MemoryType.personal,
        content=(
            "The user prefers concise, citation-first answers and avoids "
            "Wikipedia as a primary source."
        ),
    )
    assert rec is not None, "personal write returned None"

    hits = await adapter.query(
        user_id=user_id,
        project_id=project_id,
        query="What are the user's preferences for sources?",
        memory_type=MemoryType.personal,
        top_k=5,
    )
    assert len(hits) > 0, "personal retrieval returned no hits"
    assert any(
        "Wikipedia" in (h.content or "") or "concise" in (h.content or "").lower()
        for h in hits
    ), f"expected personal content in retrieval, got: {[h.content for h in hits]}"


@live_only
@pytest.mark.asyncio
async def test_reme_task_summary_then_retrieve(adapter: ReMeAdapter) -> None:
    """Write a task memory summary, then retrieve via a task-related query."""
    user_id = "live_test_user"
    project_id = "live_test"

    rec = await adapter.write(
        user_id=user_id,
        project_id=project_id,
        memory_type=MemoryType.task,
        content=(
            "When researching AWQ vs GPTQ on 70B models, prefer FP8 quantization "
            "with vLLM `--enforce-eager` to avoid torch.compile OOMs."
        ),
    )
    assert rec is not None, "task write returned None"

    hits = await adapter.query(
        user_id=user_id,
        project_id=project_id,
        query="quantization advice for 70B serving",
        memory_type=MemoryType.task,
        top_k=5,
    )
    assert len(hits) > 0, "task retrieval returned no hits"
    assert any(
        "AWQ" in (h.content or "") or "FP8" in (h.content or "") or "vLLM" in (h.content or "")
        for h in hits
    ), f"expected task content in retrieval, got: {[h.content for h in hits]}"
