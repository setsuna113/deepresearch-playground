"""ReMeAdapter — Phase 1.5 wire-up of `reme_ai.ReMeApp`.

The adapter talks to ReMe by calling `ReMeApp.async_execute(flow_name,
**kwargs)`. Flow names and payload shapes live in `reme_flows.py`.

Construction is forgiving: if `reme_ai` isn't importable, if the
embedding endpoint isn't reachable, or if `async_start()` raises for
any reason, we log a warning and leave `_reme=None`. Every read/write
method early-returns in that state, so the pipeline still runs (just
without personal/task/tool memory — working memory keeps working).

Open questions to revisit:

1. **Embedding endpoint.** ReMe needs `/v1/embeddings` from its LLM
   endpoint. vLLM serving a causal LM does NOT expose embeddings.
   For Phase 1.5 we point ReMe's embedding endpoint at the `judge`
   endpoint (typically OpenAI `text-embedding-3-small`) if available;
   otherwise we fail-soft.

2. **Tool memory retrieval.** ReMe's `retrieve_tool_memory` requires
   `tool_names`, not a free-text query. Callers must supply
   `metadata['tool_names']` (comma-separated) or we return `[]`.

3. **Tool memory writes.** ReMe expects `tool_call_results` for tool
   memory writes — a list of structured tool invocations, not the
   single reflector update we produce. We currently no-op tool writes.

4. **Inserted memory IDs.** Summary flows don't reliably surface
   inserted memory IDs in `result["metadata"]` (as of 0.3.x). We
   return a `MemoryRecord` synthesized from inputs on success.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Any

import structlog

from deepresearch.config.schema import ReMeSection
from deepresearch.memory.reme_flows import (
    RETRIEVE_FLOW_FOR,
    SUMMARY_FLOW_FOR,
    build_retrieve_kwargs,
    build_summary_trajectory,
)
from deepresearch.memory.types import reme_type_for
from deepresearch.schemas.memory import MemoryRecord, MemoryType

log = structlog.get_logger(__name__)


def _try_import_reme() -> Any | None:
    for mod in ("reme_ai", "reme"):
        try:
            return importlib.import_module(mod)
        except ImportError:
            continue
    return None


def _workspace_id(user_id: str, project_id: str) -> str:
    return f"{user_id}/{project_id}"


def _coerce_memory_records(
    raw_memories: list[Any], memory_type: MemoryType
) -> list[MemoryRecord]:
    """Convert ReMe's `BaseMemory` dicts (or instances) -> our `MemoryRecord`."""
    out: list[MemoryRecord] = []
    for m in raw_memories or []:
        # ReMeApp's `result.model_dump()` returns plain dicts; raw
        # `BaseMemory` instances are also handled defensively in case
        # the adapter is called against a stub.
        if hasattr(m, "model_dump"):
            d = m.model_dump()
        elif isinstance(m, dict):
            d = m
        else:
            log.warning("reme_unrecognized_memory_shape", value_type=type(m).__name__)
            continue
        content = _stringify(d.get("content")) or _stringify(d.get("when_to_use")) or ""
        if not content:
            continue
        score_val = d.get("score")
        try:
            score = float(score_val) if score_val is not None else None
        except (TypeError, ValueError):
            score = None
        meta: dict[str, Any] = {
            "when_to_use": d.get("when_to_use"),
            "time_created": d.get("time_created"),
            "author": d.get("author"),
        }
        out.append(
            MemoryRecord(
                id=str(d.get("memory_id") or ""),
                memory_type=memory_type,
                content=content,
                score=score,
                metadata={k: v for k, v in meta.items() if v is not None},
            )
        )
    return out


def _stringify(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8", errors="replace")
        except Exception:  # pragma: no cover - defensive
            return repr(v)
    if isinstance(v, str):
        return v
    return str(v)


def _build_reme_args(section: ReMeSection) -> tuple[list[str], dict[str, str | None]]:
    """Translate our ReMe config into CLI-style override args + constructor kwargs.

    Mirrors the `reme backend=http llm.default.model_name=... ...`
    invocation style documented in ReMeApp.__init__.

    Three known sharp edges:

    1. `vector_store.default.params={...}` is parsed as a literal string
       by flowllm's CLI parser, not a dict, so the qdrant URL override
       goes through as text and ReMe's pydantic ServiceConfig rejects
       it. We do NOT override `vector_store` here; ReMe uses its default
       in-memory store. Our working memory keeps its own Qdrant.

    2. `llm.default.model_name` is validated against a real model name.
       We read it from env (`REME_LLM_MODEL`) or fall back to
       `deepseek-chat` since DeepSeek is the most common OpenAI-compatible
       endpoint configured in this project. If a Qwen/OpenAI/etc.
       endpoint is being used, set REME_LLM_MODEL accordingly.

    3. ReMe's LLM auth: by default ReMe instantiates a plain OpenAI
       client and reads `OPENAI_API_KEY`. To use any other OpenAI-compat
       endpoint (DeepSeek, Together, hosted vLLM, etc.) we pass
       `llm_api_key` + `llm_api_base` as constructor kwargs. These
       come from `REME_LLM_API_KEY` / `REME_LLM_API_BASE` env vars,
       falling back to DEEPSEEK_API_KEY + the DeepSeek endpoint when
       a DeepSeek key is present.

    Returns (positional_args, constructor_kwargs).
    """
    # Resolve LLM credentials with the precedence:
    #   REME_LLM_API_{KEY,BASE} > DEEPSEEK_API_KEY+deepseek.com > OPENAI_API_KEY+default
    llm_api_key = os.environ.get("REME_LLM_API_KEY")
    llm_api_base = os.environ.get("REME_LLM_API_BASE")
    llm_model = os.environ.get("REME_LLM_MODEL")

    if not llm_api_key:
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        if deepseek_key:
            llm_api_key = deepseek_key
            llm_api_base = llm_api_base or "https://api.deepseek.com/v1"
            # NB: DeepSeek deprecates `deepseek-chat` / `deepseek-reasoner` on
            # 2026/07/24. `deepseek-v4-pro` is the current Pro tier and matches
            # `config.local.yaml:models.endpoints.cloud.model_id`.
            llm_model = llm_model or "deepseek-v4-pro"

    # If still nothing, fall back to OPENAI defaults — ReMe will look
    # up OPENAI_API_KEY itself if neither kwarg is supplied.
    if not llm_model:
        llm_model = "gpt-4o-mini"

    embed_api_key = os.environ.get("REME_EMBEDDING_API_KEY") or llm_api_key
    embed_api_base = os.environ.get("REME_EMBEDDING_API_BASE") or llm_api_base

    args = [
        "backend=python",  # we drive flows directly; no HTTP/MCP service
        # Suppress flowllm's Application.init_logger() from re-adding INFO
        # sinks; we own loguru config in cli/main.py.
        "init_logger=false",
        "enable_logo=false",
        "llm.default.backend=openai_compatible",
        f"llm.default.model_name={llm_model}",
        "embedding_model.default.backend=openai_compatible",
        f"embedding_model.default.model_name={section.embedding.model_id}",
        # vector_store: intentionally NOT overridden (see (1) above).
    ]
    kwargs: dict[str, str | None] = {}
    if llm_api_key:
        kwargs["llm_api_key"] = llm_api_key
    if llm_api_base:
        kwargs["llm_api_base"] = llm_api_base
    if embed_api_key:
        kwargs["embedding_api_key"] = embed_api_key
    if embed_api_base:
        kwargs["embedding_api_base"] = embed_api_base
    return args, kwargs


@dataclass
class ReMeAdapter:
    section: ReMeSection
    _reme: Any | None = None
    _stats: dict[str, int] = field(default_factory=dict)

    @classmethod
    async def create(cls, section: ReMeSection) -> ReMeAdapter:
        adapter = cls(section=section)
        if not section.enabled:
            log.info("reme_disabled_by_config")
            return adapter
        reme_mod = _try_import_reme()
        if reme_mod is None:
            log.warning("reme_not_installed", hint="`uv pip install reme-ai`")
            return adapter
        ctor = getattr(reme_mod, "ReMeApp", None) or getattr(reme_mod, "ReMe", None)
        if ctor is None:
            log.warning("reme_no_constructor", module=reme_mod.__name__)
            return adapter
        try:
            args, kwargs = _build_reme_args(section)
            app = ctor(*args, **kwargs)
            # ReMe inherits flowllm.core.application.Application; ensure
            # it's started before invoking flows.
            start = getattr(app, "async_start", None)
            if callable(start):
                await start()
            adapter._reme = app
            log.info("reme_initialized", vector_store=section.vector_store.backend)
        except Exception as e:
            log.warning("reme_init_failed", error=repr(e))
        return adapter

    @property
    def available(self) -> bool:
        return self._reme is not None

    # ---- Reads ----
    async def query(
        self,
        *,
        user_id: str,
        project_id: str,
        query: str,
        memory_type: MemoryType,
        top_k: int,
        tool_names: str | None = None,
    ) -> list[MemoryRecord]:
        if not self.available or top_k <= 0:
            return []
        flow_name = RETRIEVE_FLOW_FOR.get(memory_type)
        if flow_name is None:
            return []
        # Tool-memory retrieval requires `tool_names`; if not supplied,
        # silently skip (open question 2).
        kwargs = build_retrieve_kwargs(
            memory_type=memory_type,
            workspace_id=_workspace_id(user_id, project_id),
            query=query,
            top_k=top_k,
            tool_names=tool_names,
        )
        if kwargs is None:
            return []
        try:
            result = await self._reme.async_execute(flow_name, **kwargs)  # type: ignore[union-attr]
        except Exception as e:
            log.warning("reme_query_failed", flow=flow_name, error=repr(e))
            return []
        metadata = result.get("metadata") if isinstance(result, dict) else None
        raw_memories = (metadata or {}).get("memory_list", [])
        records = _coerce_memory_records(raw_memories, memory_type)
        self._stats[f"query_{memory_type.value}"] = self._stats.get(
            f"query_{memory_type.value}", 0
        ) + 1
        return records

    # ---- Writes ----
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
        flow_name = SUMMARY_FLOW_FOR.get(memory_type)
        if flow_name is None:
            log.warning("reme_write_unsupported", memory_type=memory_type.value)
            return None
        ws = _workspace_id(user_id, project_id)
        score = 1.0
        if metadata and isinstance(metadata.get("score"), (int, float)):
            score = float(metadata["score"])
        trajectories = build_summary_trajectory(content=content, score=score)
        try:
            result = await self._reme.async_execute(  # type: ignore[union-attr]
                flow_name,
                workspace_id=ws,
                trajectories=trajectories,
            )
        except Exception as e:
            log.warning("reme_write_failed", flow=flow_name, error=repr(e))
            return None

        # Best-effort surfaces the inserted memory_id.
        memory_id = ""
        if isinstance(result, dict):
            meta = result.get("metadata") or {}
            ids = meta.get("memory_ids") or meta.get("inserted_ids") or []
            if ids:
                memory_id = str(ids[0])
        self._stats[f"write_{memory_type.value}"] = self._stats.get(
            f"write_{memory_type.value}", 0
        ) + 1
        return MemoryRecord(
            id=memory_id,
            memory_type=memory_type,
            content=content,
            metadata=metadata or {},
        )
