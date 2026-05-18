"""ModelClient — thin async wrapper over openai.AsyncOpenAI for any
OpenAI-compatible endpoint (vLLM, SGLang, llama.cpp-server, hosted OpenAI, ...).

Records a ModelCallRecord for every call via the optional TraceRecorder.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

import structlog
from openai import AsyncOpenAI

from deepresearch.models.endpoints import EndpointSet
from deepresearch.schemas.models import Endpoint, ModelCallRecord
from deepresearch.schemas.privacy import PrivacyEnvelope

log = structlog.get_logger(__name__)


class _RecorderProtocol(Protocol):
    def record_model_call(self, rec: ModelCallRecord) -> None: ...


@dataclass
class ModelClientResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    raw: Any
    call_id: UUID


class ModelClient:
    """One client per app; calls dispatch to the right endpoint by name."""

    def __init__(
        self,
        endpoints: EndpointSet,
        recorder: _RecorderProtocol | None = None,
    ) -> None:
        self._endpoints = endpoints
        self._recorder = recorder
        self._clients: dict[str, AsyncOpenAI] = {}

    def _client_for(self, ep: Endpoint) -> AsyncOpenAI:
        if ep.name not in self._clients:
            self._clients[ep.name] = AsyncOpenAI(api_key=ep.api_key, base_url=ep.base_url)
        return self._clients[ep.name]

    async def complete(
        self,
        *,
        endpoint_name: str,
        messages: list[dict[str, Any]],
        role: str,
        run_id: UUID | None = None,
        step_id: UUID | None = None,
        envelope: PrivacyEnvelope | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ModelClientResponse:
        ep = self._endpoints.get(endpoint_name)
        client = self._client_for(ep)
        kwargs: dict[str, Any] = {
            "model": ep.model_id,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format
        if tools is not None:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice

        t0 = time.perf_counter()
        error: str | None = None
        try:
            resp = await client.chat.completions.create(**kwargs)  # type: ignore[arg-type]
            _msg = resp.choices[0].message
            text = _msg.content or ""
            # DeepSeek's reasoning models (v4-flash, v4-pro) put their
            # actual answer text under `reasoning_content` when the
            # output budget is small or the model decides to stay in
            # reasoning mode. Fall back so the caller sees SOMETHING
            # rather than an empty string that downstream nodes treat
            # as "no report".
            if not text:
                text = getattr(_msg, "reasoning_content", "") or ""
            usage = resp.usage
            prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
            completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        except Exception as e:
            error = repr(e)
            raise
        finally:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            rec = ModelCallRecord(
                run_id=run_id,
                step_id=step_id,
                endpoint_name=ep.name,
                model_id=ep.model_id,
                role=role,
                prompt_tokens=locals().get("prompt_tokens", 0),
                completion_tokens=locals().get("completion_tokens", 0),
                latency_ms=latency_ms,
                envelope=envelope or PrivacyEnvelope.default_public(),
                error=error,
            )
            if self._recorder is not None:
                self._recorder.record_model_call(rec)
            if error:
                log.warning("model_call_error", endpoint=ep.name, role=role, error=error)
        return ModelClientResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw=resp,
            call_id=rec.id,
        )
