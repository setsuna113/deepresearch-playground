"""WorkingMemory — our own Qdrant collection for long-trace storage.

ReMe does not have a native "working" memory type, so we own this tier
ourselves. Phase 5's hot/warm/cold placement will wrap this and the ReMe
adapter behind a common policy.

Embedding strategy:
- `embedding_model == "hash-fallback"`: deterministic SHA-256-derived
  vector. Semantic-blind, useful only for hermetic tests where we don't
  care about retrieval quality.
- otherwise: `sentence_transformers.SentenceTransformer(model_id,
  device="cpu")`. The default in `config.example.yaml` is
  `BAAI/bge-small-en-v1.5` (~140 MB, 384-dim) — small enough to run on
  CPU without latency surprises, large enough to support real
  retrieval.

The collection's vector size is set to match the embedder on first
write. Switching models mid-run requires deleting the on-disk Qdrant
collection (the dim won't match).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

from deepresearch.config.schema import WorkingMemoryConfig
from deepresearch.schemas.memory import MemoryRecord, MemoryType

log = structlog.get_logger(__name__)

_HASH_VECTOR_SIZE = 384  # match bge-small-en-v1.5 so swapping models
# between hash-fallback and the real model uses the same collection
# dim — avoids a Qdrant mismatch the first time a dev switches over.


def _hash_embedding(text: str, size: int = _HASH_VECTOR_SIZE) -> list[float]:
    """Deterministic fallback embedding. NOT a substitute for a real
    semantic encoder — used only when `embedding_model == "hash-fallback"`
    for hermetic tests."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    raw = (h * ((size // len(h)) + 1))[:size]
    return [(b - 128) / 128.0 for b in raw]


@dataclass
class _Embedder:
    """Lazy-loaded encoder. The sentence-transformers model is only
    loaded on first non-hash use, so test runs that stick to
    `hash-fallback` never pull torch into memory."""

    model_id: str
    _dim: int | None = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)

    @property
    def dim(self) -> int:
        if self._dim is None:
            if self.model_id == "hash-fallback":
                self._dim = _HASH_VECTOR_SIZE
            else:
                self._ensure_loaded()
        return int(self._dim or 0)

    def _ensure_loaded(self) -> None:
        if self.model_id == "hash-fallback" or self._model is not None:
            return
        # Imported lazily so the dependency only impacts processes that
        # actually exercise it (tests on hash-fallback skip the import).
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_id, device="cpu")
        # sentence-transformers >=5.0 renamed get_sentence_embedding_dimension
        # → get_embedding_dimension. Support both.
        dim_fn = getattr(
            self._model, "get_embedding_dimension", None
        ) or self._model.get_sentence_embedding_dimension
        self._dim = int(dim_fn())
        log.info(
            "working_memory_encoder_loaded",
            model_id=self.model_id,
            dim=self._dim,
        )

    def encode(self, text: str) -> list[float]:
        if self.model_id == "hash-fallback":
            return _hash_embedding(text, _HASH_VECTOR_SIZE)
        self._ensure_loaded()
        assert self._model is not None
        vec = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return [float(x) for x in vec.tolist()]


@dataclass
class WorkingMemory:
    cfg: WorkingMemoryConfig
    client: AsyncQdrantClient
    embedder: _Embedder

    @classmethod
    async def create(cls, cfg: WorkingMemoryConfig) -> WorkingMemory:
        if cfg.local_path:
            Path(cfg.local_path).mkdir(parents=True, exist_ok=True)
            client = AsyncQdrantClient(path=cfg.local_path)
            log.info("working_memory_local_mode", path=cfg.local_path)
        else:
            client = AsyncQdrantClient(url=cfg.qdrant_url)
            log.info("working_memory_server_mode", url=cfg.qdrant_url)
        embedder = _Embedder(cfg.embedding_model)
        return cls(cfg=cfg, client=client, embedder=embedder)

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
                vectors_config=qmodels.VectorParams(
                    size=self.embedder.dim, distance=qmodels.Distance.COSINE
                ),
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
        vec = self.embedder.encode(content)
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
            vec = self.embedder.encode(query)
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
