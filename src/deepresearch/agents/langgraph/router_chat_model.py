"""`RouterChatModel` — the LangGraph-facing chat model that routes every
call through `Router.select()`.

The vendored `open_deep_research` builds a single module-level
`configurable_model = init_chat_model(configurable_fields=("model",
"max_tokens", "api_key"))` and then per-node calls
`.with_structured_output(...).with_retry(...).with_config({"model":
"<role>", "max_tokens": ..., "api_key": "..."})`. We replace that with
`build_router_configurable_model(deps, request, run_id)` which returns
an object behaviorally compatible with `_ConfigurableModel`:

- Captures declarative ops (`with_structured_output`, `with_retry`,
  `bind_tools`) into a queue.
- Stores the model/max_tokens/api_key values fed via `with_config(...)`.
- On `ainvoke(...)` materializes a `RouterChatModel(role=<model>, ...)`
  with the queued ops applied, then calls into it.

This way **every** LLM call from any LangGraph node dispatches through
our async `ModelClient.complete()` via `Router.select(profile, role,
envelope, hint)` — preserving the Phase-4 ParetoDispatch seam.

The role string is whatever upstream's `Configuration.research_model`
(and the other three `*_model` fields) was set to. The `runtime.py`
layer sets these to our role names: "supervisor", "compressor",
"final_report" — see `role_map.CONFIG_FIELD_TO_ROLE`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import structlog
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict, Field

from deepresearch.agents.context import RunDependencies
from deepresearch.schemas.privacy import PrivacyEnvelope
from deepresearch.schemas.runs import RunRequest

log = structlog.get_logger(__name__)


# Declarative methods we queue (they live on BaseChatModel, not on
# Runnable, so our proxy must intercept them). `with_retry`,
# `with_fallbacks`, and similar are inherited from Runnable and wrap
# the proxy with their own Runnable variants — that's fine; when those
# wrappers eventually call our `ainvoke`, materialization still picks
# up the queued ops below.
_FORWARDED_DECLARATIVE = frozenset({"with_structured_output", "bind_tools"})


def _messages_to_openai(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """Translate LangChain messages -> OpenAI chat-completions dicts.

    Covers the four message types open_deep_research actually emits.
    Tool messages preserve their `tool_call_id`. Assistant messages with
    tool calls pass through with the `tool_calls` array intact.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": _text(m.content)})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": _text(m.content)})
        elif isinstance(m, ToolMessage):
            out.append(
                {
                    "role": "tool",
                    "content": _text(m.content),
                    "tool_call_id": m.tool_call_id,
                }
            )
        elif isinstance(m, AIMessage):
            entry: dict[str, Any] = {"role": "assistant", "content": _text(m.content)}
            if m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": _json_dumps(tc["args"]),
                        },
                    }
                    for tc in m.tool_calls
                ]
            out.append(entry)
        else:
            # Fallback for unknown message types — render as user text so we
            # never silently drop content.
            out.append({"role": "user", "content": _text(getattr(m, "content", str(m)))})
    return out


def _text(content: Any) -> str:
    """LangChain message content can be str or a list of content parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def _json_dumps(obj: Any) -> str:
    import json

    if isinstance(obj, str):
        return obj
    return json.dumps(obj, ensure_ascii=False)


def _parse_tool_calls(raw_tool_calls: Any) -> list[dict[str, Any]]:
    """Convert OpenAI-format tool_calls -> LangChain ToolCall dicts."""
    import json

    out: list[dict[str, Any]] = []
    for tc in raw_tool_calls or []:
        fn = tc.function if hasattr(tc, "function") else tc.get("function", {})
        name = fn.name if hasattr(fn, "name") else fn.get("name", "")
        args_str = fn.arguments if hasattr(fn, "arguments") else fn.get("arguments", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            args = {"_raw": args_str}
        out.append(
            {
                "id": tc.id if hasattr(tc, "id") else tc.get("id", ""),
                "name": name,
                "args": args,
                "type": "tool_call",
            }
        )
    return out


class RouterChatModel(BaseChatModel):
    """Concrete `BaseChatModel` that dispatches through `Router.select()`.

    Constructed by `RouterConfigurableModel._materialize()` once the
    declarative chain has supplied a `role` via `with_config({"model":
    role})`. Carries the full execution context needed by
    `ModelClient.complete` for trace attribution.

    NOTE: `deps`, `envelope`, and other non-pydantic objects are declared
    with `Any` so Pydantic v2 doesn't walk their (sometimes-dataclass)
    internals — that introspection breaks on nested dataclass-of-dataclass
    chains in our `RunDependencies` graph.
    """

    # Typed as Any to keep Pydantic out of our dependency dataclasses.
    deps: Any
    profile_name: str
    envelope: Any = None
    run_id: Any = None
    role: str = "supervisor"
    max_tokens_override: int | None = None
    # `bound_tools` carries OpenAI-formatted tool schemas. Captured from
    # `bind_tools(...)` calls on the configurable proxy.
    bound_tools: list[dict[str, Any]] = Field(default_factory=list)
    bound_tool_choice: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return "deepresearch-router-chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError(
            "RouterChatModel is async-only; use .ainvoke() or .agenerate()."
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        ep = self.deps.router.select(
            profile=self.profile_name,
            role=self.role,
            envelope=self.envelope,
            hint=None,
        )
        oai_messages = _messages_to_openai(messages)

        complete_kwargs: dict[str, Any] = {
            "endpoint_name": ep.name,
            "messages": oai_messages,
            "role": self.role,
            "run_id": self.run_id,
            "envelope": self.envelope,
            "temperature": float(kwargs.get("temperature", 0.3)),
        }
        if self.max_tokens_override is not None:
            complete_kwargs["max_tokens"] = self.max_tokens_override
        elif "max_tokens" in kwargs:
            complete_kwargs["max_tokens"] = kwargs["max_tokens"]

        # Tool-calling: prefer explicit kwargs override, else use bound tools.
        tools_to_send = kwargs.get("tools") or self.bound_tools
        if tools_to_send:
            complete_kwargs["tools"] = tools_to_send
            tool_choice = kwargs.get("tool_choice") or self.bound_tool_choice or "auto"
            complete_kwargs["tool_choice"] = tool_choice

        # Structured output: open_deep_research uses with_structured_output(...)
        # which the configurable proxy translates into a JSON-mode hint
        # via kwargs. The actual schema validation happens via the
        # PydanticOutputParser appended to the runnable chain.
        if kwargs.get("response_format"):
            complete_kwargs["response_format"] = kwargs["response_format"]

        resp = await self.deps.model_client.complete(**complete_kwargs)

        # Build AIMessage from the raw OpenAI response so LangChain's
        # ToolMessage parsing, content streaming, etc. all work.
        msg = resp.raw.choices[0].message
        ai = AIMessage(
            content=resp.text,
            tool_calls=_parse_tool_calls(getattr(msg, "tool_calls", None)),
            usage_metadata=UsageMetadata(
                input_tokens=resp.prompt_tokens,
                output_tokens=resp.completion_tokens,
                total_tokens=resp.prompt_tokens + resp.completion_tokens,
            ),
            response_metadata={
                "model_id": ep.model_id,
                "endpoint": ep.name,
                "call_id": str(resp.call_id),
            },
        )
        return ChatResult(
            generations=[ChatGeneration(message=ai, generation_info={"call_id": str(resp.call_id)})]
        )

    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> RouterChatModel:
        formatted = [convert_to_openai_tool(t) for t in tools]
        return self.model_copy(
            update={"bound_tools": formatted, "bound_tool_choice": tool_choice}
        )


class RouterConfigurableModel(Runnable[LanguageModelInput, AIMessage]):
    """Behavioral clone of LangChain's `_ConfigurableModel` that defers
    model materialization to our `RouterChatModel`.

    The vendored `open_deep_research` constructs this once at module
    load (`configurable_model = init_chat_model(configurable_fields=...)`)
    and threads it through every node. Each node calls a chain of
    declarative ops then `with_config({"model": role, "max_tokens": N,
    "api_key": ...})` to pick a model — for us, the "model" string is
    the role name and we ignore api_key (the endpoint carries it).
    """

    def __init__(
        self,
        *,
        deps: RunDependencies,
        profile_name: str,
        envelope: PrivacyEnvelope,
        run_id: UUID | None,
        default_config: dict[str, Any] | None = None,
        queued_ops: list[tuple[str, tuple, dict]] | None = None,
    ) -> None:
        self._deps = deps
        self._profile_name = profile_name
        self._envelope = envelope
        self._run_id = run_id
        self._default_config: dict[str, Any] = dict(default_config or {})
        self._queued_ops: list[tuple[str, tuple, dict]] = list(queued_ops or [])

    def _clone(
        self,
        *,
        default_config: dict[str, Any] | None = None,
        queued_ops: list[tuple[str, tuple, dict]] | None = None,
    ) -> RouterConfigurableModel:
        return RouterConfigurableModel(
            deps=self._deps,
            profile_name=self._profile_name,
            envelope=self._envelope,
            run_id=self._run_id,
            default_config=default_config
            if default_config is not None
            else dict(self._default_config),
            queued_ops=queued_ops
            if queued_ops is not None
            else list(self._queued_ops),
        )

    # -- Declarative ops queue (mirrors _ConfigurableModel.__getattr__) --
    def __getattr__(self, name: str) -> Any:
        if name in _FORWARDED_DECLARATIVE:
            def queue(*args: Any, **kwargs: Any) -> RouterConfigurableModel:
                new_ops = [*self._queued_ops, (name, args, kwargs)]
                return self._clone(queued_ops=new_ops)

            return queue
        raise AttributeError(f"{name!r} is not supported on RouterConfigurableModel")

    # -- Config absorption (mirrors _ConfigurableModel.with_config) --
    def with_config(
        self,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> RouterConfigurableModel:
        merged: dict[str, Any] = dict(config or {})
        merged.update(kwargs)
        # Upstream's per-node `model_config` is passed at the top level
        # (not under "configurable"): {"model": ..., "max_tokens": ...,
        # "api_key": ..., "tags": [...]}. Pull out the model/max_tokens
        # keys we care about and put everything else back as a
        # queued with_config so callback metadata still flows.
        absorbed_keys = {"model", "max_tokens", "api_key"}
        new_default = dict(self._default_config)
        for k in list(merged.keys()):
            if k in absorbed_keys:
                new_default[k] = merged.pop(k)
        # Also accept the LangChain-canonical form where these live
        # under "configurable".
        for k in list(merged.get("configurable", {}).keys()):
            if k in absorbed_keys:
                new_default[k] = merged["configurable"].pop(k)

        new_ops = list(self._queued_ops)
        if merged:
            # Preserve tags / callbacks / metadata via a queued with_config.
            new_ops.append(("with_config", (), {"config": merged}))
        return self._clone(default_config=new_default, queued_ops=new_ops)

    # -- Materialization --
    def _materialize(self, runtime_config: RunnableConfig | None = None) -> Runnable:
        # Merge our captured default_config with anything LangChain
        # promoted into `runtime_config["configurable"]` (e.g., via a
        # `RunnableBinding.with_config` upstream of us).
        cfg: dict[str, Any] = dict(self._default_config)
        if runtime_config is not None:
            configurable = runtime_config.get("configurable") or {}
            for k in ("model", "max_tokens", "api_key"):
                if k in configurable and k not in cfg:
                    cfg[k] = configurable[k]
        role = cfg.get("model")
        if not role:
            raise RuntimeError(
                "RouterChatModel requires a role; none was supplied via "
                ".with_config({'model': '<role>'})"
            )
        max_tokens = cfg.get("max_tokens")
        model: Runnable = RouterChatModel(
            deps=self._deps,
            profile_name=self._profile_name,
            envelope=self._envelope,
            run_id=self._run_id,
            role=role,
            max_tokens_override=max_tokens,
        )
        for name, args, kwargs in self._queued_ops:
            model = getattr(model, name)(*args, **kwargs)
        return model

    # -- Runnable protocol --
    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        materialized = self._materialize(config)
        return await materialized.ainvoke(input, config=config, **kwargs)

    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        materialized = self._materialize(config)
        return materialized.invoke(input, config=config, **kwargs)

    async def astream(  # pragma: no cover - upstream uses ainvoke, not astream
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        materialized = self._materialize(config)
        async for chunk in materialized.astream(input, config=config, **kwargs):
            yield chunk

    def stream(  # pragma: no cover - sync streaming not used
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        materialized = self._materialize(config)
        yield from materialized.stream(input, config=config, **kwargs)


def build_router_configurable_model(
    *,
    deps: RunDependencies,
    request: RunRequest,
    run_id: UUID | None,
) -> RouterConfigurableModel:
    """Construct a configurable proxy for a single research run.

    Replaces the module-level `configurable_model = init_chat_model(...)`
    call inside the vendored `deep_researcher.py` (see UPSTREAM_NOTE.md
    "Patch 2: model factory hook").
    """
    return RouterConfigurableModel(
        deps=deps,
        profile_name=request.model_profile,
        envelope=request.privacy_envelope
        if request.privacy_envelope is not None
        else PrivacyEnvelope.default_public(),
        run_id=run_id,
    )


# ----------------------------------------------------------------------
# ContextVar-based active-run mechanism.
#
# The vendored open_deep_research keeps `configurable_model` as a
# module-level singleton. To inject our `RouterConfigurableModel` per
# run (with run-specific deps + privacy envelope) without modifying
# every upstream node function, we provide a lazy proxy whose attribute
# accesses delegate to a contextvar-bound `RouterConfigurableModel`.
#
# `runtime.run_research` enters `active_run_context(...)` before
# invoking the graph; the proxy then sees the active instance. Since
# ContextVars propagate through asyncio tasks (including those spawned
# by `asyncio.gather` for parallel researchers), concurrent runs are
# safe as long as they each enter their own context.
# ----------------------------------------------------------------------


@dataclass
class _ActiveRun:
    deps: Any
    request: Any
    run_id: Any


_active_run_var: ContextVar[_ActiveRun | None] = ContextVar(
    "deepresearch_active_run", default=None
)


def get_active_router_model() -> RouterConfigurableModel:
    """Return a fresh `RouterConfigurableModel` for the active run.

    Raises if no `active_run_context(...)` is in scope — meaning the
    vendored graph was invoked without runtime.py setting up the context,
    which is a developer error.
    """
    active = _active_run_var.get()
    if active is None:
        raise RuntimeError(
            "RouterChatModel: no active run context. Wrap the graph "
            "invocation with `active_run_context(deps, request, run_id)`."
        )
    return build_router_configurable_model(
        deps=active.deps, request=active.request, run_id=active.run_id
    )


@contextmanager
def active_run_context(*, deps: Any, request: Any, run_id: Any):
    """Bind (deps, request, run_id) for the duration of a graph invocation."""
    token = _active_run_var.set(_ActiveRun(deps=deps, request=request, run_id=run_id))
    try:
        yield
    finally:
        _active_run_var.reset(token)


class _LazyConfigurableModelProxy:
    """Module-level stand-in for the vendored `configurable_model`.

    Every attribute access resolves the active `RouterConfigurableModel`
    via the contextvar and delegates. Forwards the small subset of
    methods upstream actually calls: `with_config`, `with_structured_output`,
    `with_retry`, `bind_tools`, `with_fallbacks`, `ainvoke`, `invoke`,
    `astream`, `stream`. AttributeError on anything else.
    """

    def __getattr__(self, name: str) -> Any:
        return getattr(get_active_router_model(), name)


# Singleton expected by the vendored module-level
# `configurable_model = init_chat_model(...)` replacement.
configurable_model_proxy = _LazyConfigurableModelProxy()


__all__ = [
    "RouterChatModel",
    "RouterConfigurableModel",
    "active_run_context",
    "build_router_configurable_model",
    "configurable_model_proxy",
    "get_active_router_model",
]
