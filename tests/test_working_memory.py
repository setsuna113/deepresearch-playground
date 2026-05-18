"""Working memory roundtrip — uses qdrant-client embedded mode so no
docker / Qdrant server is needed.

The basic roundtrip uses `hash-fallback` so tests don't pay the
SentenceTransformer download cost. A separate semantic test uses the
real `BAAI/bge-small-en-v1.5` model and asserts that related queries
score higher than unrelated ones — guarding against silent regressions
where the encoder is mis-wired (e.g. all-zeros vectors)."""

from __future__ import annotations

import pytest

from deepresearch.config.schema import WorkingMemoryConfig
from deepresearch.memory.working_memory import WorkingMemory


@pytest.mark.asyncio
async def test_working_memory_local_roundtrip(tmp_path):
    cfg = WorkingMemoryConfig(
        local_path=str(tmp_path / "qdrant"),
        collection_template="dr_working_{user}_{project}",
        embedding_model="hash-fallback",
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


@pytest.mark.asyncio
async def test_working_memory_real_embeddings_are_semantic(tmp_path):
    """With a real sentence-transformers encoder, queries about
    semantically related concepts should score higher than queries about
    unrelated concepts. The hash-fallback would fail this — its scores
    are decoupled from meaning, so this test is the canary for the
    semantic-blind regression."""
    cfg = WorkingMemoryConfig(
        local_path=str(tmp_path / "qdrant"),
        collection_template="dr_working_{user}_{project}",
        embedding_model="BAAI/bge-small-en-v1.5",
    )
    wm = await WorkingMemory.create(cfg)

    # Seed two distinct topical memories.
    await wm.write(
        user_id="alice",
        project_id="thesis",
        content="Transformer architectures dominate modern NLP",
    )
    await wm.write(
        user_id="alice",
        project_id="thesis",
        content="Sourdough bread needs 70% hydration and a wild yeast starter",
    )

    # A semantically related query should retrieve the NLP memory with
    # a notably higher score than the baking memory.
    hits = await wm.query(
        user_id="alice",
        project_id="thesis",
        query="deep learning language models",
        top_k=2,
    )
    assert len(hits) == 2
    # Top hit should be the NLP memory.
    top = hits[0]
    assert "Transformer" in top.content
    assert top.score is not None and top.score > 0.3, (
        f"semantic encoder should produce a meaningful score, got {top.score}"
    )
    # The unrelated baking memory should score lower than the NLP one.
    nlp_score = next(
        (h.score for h in hits if "Transformer" in h.content), None
    )
    bread_score = next(
        (h.score for h in hits if "Sourdough" in h.content), None
    )
    assert nlp_score is not None and bread_score is not None
    assert nlp_score > bread_score, (
        f"NLP query should outrank bread memory; got nlp={nlp_score} bread={bread_score}"
    )
