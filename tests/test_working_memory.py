"""Working memory roundtrip — uses qdrant-client embedded mode so no
docker / Qdrant server is needed."""

from __future__ import annotations

import pytest

from deepresearch.config.schema import WorkingMemoryConfig
from deepresearch.memory.working_memory import WorkingMemory


@pytest.mark.asyncio
async def test_working_memory_local_roundtrip(tmp_path):
    cfg = WorkingMemoryConfig(
        local_path=str(tmp_path / "qdrant"),
        collection_template="dr_working_{user}_{project}",
    )
    wm = await WorkingMemory.create(cfg)
    rec = await wm.write(
        user_id="alice",
        project_id="thesis",
        content="user dislikes Wikipedia as primary source",
    )
    assert rec.content.startswith("user dislikes")

    hits = await wm.query(
        user_id="alice", project_id="thesis", query="Wikipedia preference", top_k=3
    )
    assert any("Wikipedia" in h.content for h in hits)
