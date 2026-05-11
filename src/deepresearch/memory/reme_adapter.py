"""ReMeAdapter — async wrapper over ReMe's Python API.

The ReMe package (`reme-ai` on PyPI, verified against 0.3.1.8) exposes
its main class as `reme_ai.ReMeApp` — a subclass of `flowllm.core.application.Application`.
Both `reme-ai` and `flowllm` are now pinned in pyproject.toml; the
sibling `reme` package would additionally require `agentscope`, which we
don't pull in.

Wire-up TODO (Phase 1.5): the ReMeApp API is a flow-driven Application —
calls are made by invoking named "flows" with structured payloads rather
than direct `.search`/`.add` methods. The pseudocode below is best-effort
based on the upstream README's vocabulary; verify against a live install
before relying on retrieve/write semantics.

Everything outside this file speaks `MemoryRecord` / `MemoryType`; only
this adapter touches ReMe.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import structlog

from deepresearch.config.schema import ReMeSection
from deepresearch.memory.types import reme_type_for
from deepresearch.schemas.memory import MemoryRecord, MemoryType

log = structlog.get_logger(__name__)


def _try_import_reme() -> Any | None:
    # Prefer `reme_ai` (needs flowllm only) over `reme` (needs agentscope).
    for mod in ("reme_ai", "reme"):
        try:
            return importlib.import_module(mod)
        except ImportError:
            continue
    return None


@dataclass
class ReMeAdapter:
    section: ReMeSection
    _reme: Any | None = None

    @classmethod
    async def create(cls, section: ReMeSection) -> "ReMeAdapter":
        adapter = cls(section=section)
        if not section.enabled:
            return adapter
        reme_mod = _try_import_reme()
        if reme_mod is None:
            log.warning("reme_not_installed", hint="install with: uv pip install reme-ai")
            return adapter
        try:
            # Best-effort init — exact signature confirmed when first wired live.
            cfg = {
                "vector_store": section.vector_store.model_dump(),
                "llm": section.llm.model_dump(),
                "embedding": section.embedding.model_dump(),
                "working_dir": section.working_dir,
            }
            ctor = (
                getattr(reme_mod, "ReMeApp", None)
                or getattr(reme_mod, "ReMe", None)
                or getattr(reme_mod, "ReMeLight", None)
            )
            if ctor is None:
                log.warning("reme_no_constructor", module=reme_mod.__name__)
                return adapter
            # ReMeApp expects llm_api_*, embedding_api_*, config_path — NOT
            # the cfg dict above. Phase 1.5 will wire this properly.
            log.warning(
                "reme_init_skipped_phase1",
                hint="ReMeApp wire-up pending; using working memory only",
            )
            _ = ctor, cfg
            return adapter
        except Exception as e:
            log.warning("reme_init_failed", error=repr(e))
        return adapter

    @property
    def available(self) -> bool:
        return self._reme is not None

    async def query(
        self,
        *,
        user_id: str,
        project_id: str,
        query: str,
        memory_type: MemoryType,
        top_k: int,
    ) -> list[MemoryRecord]:
        if not self.available or top_k <= 0:
            return []
        reme_t = reme_type_for(memory_type)
        if reme_t is None:
            return []
        try:
            # ReMe API (confirm against installed version):
            #   await self._reme.search(workspace=..., query=..., memory_type=..., top_k=...)
            results = await self._reme.search(  # type: ignore[union-attr]
                workspace=f"{user_id}/{project_id}",
                query=query,
                memory_type=reme_t,
                top_k=top_k,
            )
        except Exception as e:
            log.warning("reme_query_failed", error=repr(e), memory_type=memory_type)
            return []
        out: list[MemoryRecord] = []
        for r in results or []:
            out.append(
                MemoryRecord(
                    id=str(r.get("id") or r.get("memory_id") or ""),
                    memory_type=memory_type,
                    content=str(r.get("content") or r.get("text") or ""),
                    score=float(r.get("score")) if r.get("score") is not None else None,
                    metadata=dict(r.get("metadata") or {}),
                )
            )
        return out

    async def write(
        self,
        *,
        user_id: str,
        project_id: str,
        memory_type: MemoryType,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord | None:
        if not self.available:
            return None
        reme_t = reme_type_for(memory_type)
        if reme_t is None:
            return None
        try:
            r = await self._reme.add(  # type: ignore[union-attr]
                workspace=f"{user_id}/{project_id}",
                content=content,
                memory_type=reme_t,
                metadata=metadata or {},
            )
        except Exception as e:
            log.warning("reme_write_failed", error=repr(e), memory_type=memory_type)
            return None
        return MemoryRecord(
            id=str(r.get("id") if isinstance(r, dict) else r or ""),
            memory_type=memory_type,
            content=content,
            metadata=metadata or {},
        )
