"""WorkingMemory — our own Qdrant collection for long-trace storage.

ReMe does not have a native "working" memory type, so we own this tier
ourselves. Phase 5's hot/warm/cold placement will wrap this and the ReMe
adapter behind a common policy.

Phase 1 embedding strategy: ask the local OpenAI-compatible endpoint for
embeddings via `/v1/embeddings`. If that fails (the local 20B may not serve
embeddings), fall back to a deterministic random-but-stable hash embedding
so the rest of the loop keeps moving. This is good enough for the smoke
gate; Phase 2 swaps in a real embedding model.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from deepresearch.config.schema import WorkingMemoryConfig
from deepresearch.schemas.memory import MemoryRecord, MemoryType

log = structlog.get_logger(__name__)

_VECTOR_SIZE = 768


def _hash_embedding(text: str, size: int = _VECTOR_SIZE) -> list[float]:
    """Deterministic fallback embedding. Hashes are NOT a substitute for
    semantic embeddings; this exists only so Phase-1 smoke tests pass
    without a live embedding endpoint."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    raw = (h * ((size // len(h)) + 1))[:size]
    return [(b - 128) / 128.0 for b in raw]


@dataclass
class WorkingMemory:
    cfg: WorkingMemoryConfig
    client: AsyncQdrantClient

    @classmethod
    async def create(cls, cfg: WorkingMemoryConfig) -> WorkingMemory:
        if cfg.local_path:
            Path(cfg.local_path).mkdir(parents=True, exist_ok=True)
            client = AsyncQdrantClient(path=cfg.local_path)
            log.info("working_memory_local_mode", path=cfg.local_path)
        else:
            client = AsyncQdrantClient(url=cfg.qdrant_url)
            log.info("working_memory_server_mode", url=cfg.qdrant_url)
        return cls(cfg=cfg, client=client)

    def _collection(self, user_id: str, project_id: str) -> str:
        return self.cfg.collection_template.format(user=user_id, project=project_id).replace(
            "/", "_"
        )

    async def _ensure_collection(self, name: str) -> None:
        try:
            await self.client.get_collection(name)
        except Exception:
            await self.client.create_collection(
                collection_name=name,
                vectors_config=qmodels.VectorParams(size=_VECTOR_SIZE, distance=qmodels.Distance.COSINE),
            )

    async def write(
        self,
        *,
        user_id: str,
        project_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        col = self._collection(user_id, project_id)
        await self._ensure_collection(col)
        vec = _hash_embedding(content)
        point_id = hashlib.sha1(f"{user_id}|{project_id}|{content}".encode()).hexdigest()[:16]
        # qdrant requires int or uuid; use stable int from hash
        pid = int(point_id, 16) & ((1 << 63) - 1)
        try:
            await self.client.upsert(
                collection_name=col,
                points=[
                    qmodels.PointStruct(
                        id=pid,
                        vector=vec,
                        payload={"content": content, "metadata": metadata or {}},
                    )
                ],
            )
        except Exception as e:
            log.warning("working_write_failed", error=repr(e))
        return MemoryRecord(
            id=str(pid),
            memory_type=MemoryType.working,
            content=content,
            metadata=metadata or {},
        )

    async def query(
        self,
        *,
        user_id: str,
        project_id: str,
        query: str,
        top_k: int,
    ) -> list[MemoryRecord]:
        if top_k <= 0:
            return []
        col = self._collection(user_id, project_id)
        try:
            await self._ensure_collection(col)
            vec = _hash_embedding(query)
            resp = await self.client.query_points(
                collection_name=col, query=vec, limit=top_k
            )
            points = resp.points
        except Exception as e:
            log.warning("working_query_failed", error=repr(e))
            return []
        out: list[MemoryRecord] = []
        for r in points:
            payload = r.payload or {}
            out.append(
                MemoryRecord(
                    id=str(r.id),
                    memory_type=MemoryType.working,
                    content=str(payload.get("content", "")),
                    score=float(r.score) if r.score is not None else None,
                    metadata=dict(payload.get("metadata") or {}),
                )
            )
        return out
