"""Unit tests for `RouterChatModel` and the `RouterConfigurableModel` proxy.

We exercise the seam end-to-end without hitting any real LLM endpoint by
substituting a fake `ModelClient` that captures every `.complete(...)`
call. The real `Router` + `EndpointSet` resolve the endpoint name so we
also verify the profile-keyed dispatch path that Phase-4 will replace.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from deepresearch.agents.context import RunDependencies
from deepresearch.agents.langgraph.router_chat_model import (
    RouterConfigurableModel,
    build_router_configurable_model,
)
from deepresearch.config.schema import EndpointConfig, ModelProfileConfig, ModelsSection
from deepresearch.models.client import ModelClientResponse
from deepresearch.models.endpoints import EndpointSet
from deepresearch.models.router import Router
from deepresearch.schemas.privacy import PrivacyEnvelope
from deepresearch.schemas.runs import RunRequest

# ---- Test doubles --------------------------------------------------------


class _FakeModelClient:
    """Captures every .complete() call and returns a canned response."""

    def __init__(self, *, response_text: str = "ok", tool_calls=None) -> None:
        self.calls: list[dict] = []
        self._response_text = response_text
        self._tool_calls = tool_calls or []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        # Mimic the OpenAI SDK shape ModelClient parses (resp.raw.choices[0].message).
        msg = SimpleNamespace(content=self._response_text, tool_calls=self._tool_calls or None)
        raw = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
        return ModelClientResponse(
            text=self._response_text,
            prompt_tokens=12,
            completion_tokens=7,
            raw=raw,
            call_id=uuid4(),
        )


def _make_endpoints() -> EndpointSet:
    section = ModelsSection(
        endpoints={
            "local": EndpointConfig(
                base_url="http://localhost:8001/v1",
                api_key="EMPTY",
                model_id="qwen3-8b-awq",
            ),
            "cloud": EndpointConfig(
                base_url="http://localhost:8002/v1",
                api_key="EMPTY",
                model_id="qwen2.5-72b-instruct-awq",
            ),
        },
        profiles={
            "phase1_default": ModelProfileConfig(
                planner="local",
                searcher="local",
                reader="local",
                synthesizer="local",
                reflector="local",
            ),
            "co_schedule_v0": ModelProfileConfig(
                planner="local",
                searcher="local",
                reader="cloud",
                synthesizer="cloud",
                reflector="local",
            ),
        },
    )
    return EndpointSet.from_config(section)


def _make_deps(client: _FakeModelClient) -> RunDependencies:
    endpoints = _make_endpoints()
    router = Router(endpoints)
    return RunDependencies(
        config=SimpleNamespace(),  # type: ignore[arg-type]
        repos=SimpleNamespace(),  # type: ignore[arg-type]
        recorder=SimpleNamespace(record_step=lambda *_a, **_k: None),  # type: ignore[arg-type]
        model_client=client,  # type: ignore[arg-type]
        router=router,
        memory=SimpleNamespace(),  # type: ignore[arg-type]
        tools=SimpleNamespace(),  # type: ignore[arg-type]
    )


def _make_request(profile: str = "phase1_default") -> RunRequest:
    return RunRequest(
        query="test query",
        user_id="alice",
        project_id="thesis",
        model_profile=profile,
        privacy_envelope=PrivacyEnvelope.default_public(),
    )


# ---- Tests ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_chat_model_routes_via_router_select() -> None:
    client = _FakeModelClient(response_text="hello world")
    deps = _make_deps(client)
    req = _make_request(profile="phase1_default")

    proxy = build_router_configurable_model(deps=deps, request=req, run_id=uuid4())
    chain = proxy.with_config({"model": "planner", "max_tokens": 1024, "api_key": "ignored"})
    msg = await chain.ainvoke([HumanMessage(content="hi")])

    assert msg.content == "hello world"
    assert msg.usage_metadata["input_tokens"] == 12
    assert msg.usage_metadata["output_tokens"] == 7
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["endpoint_name"] == "local"
    assert call["role"] == "planner"
    assert call["max_tokens"] == 1024
    # The role string is supplied via with_config; api_key is absorbed
    # and dropped (we use the endpoint's own key).
    assert "api_key" not in call


@pytest.mark.asyncio
async def test_router_chat_model_routes_to_cloud_for_cloud_role() -> None:
    client = _FakeModelClient()
    deps = _make_deps(client)
    req = _make_request(profile="co_schedule_v0")

    proxy = build_router_configurable_model(deps=deps, request=req, run_id=uuid4())
    chain = proxy.with_config({"model": "synthesizer", "max_tokens": 512})
    await chain.ainvoke([HumanMessage(content="hi")])

    assert client.calls[0]["endpoint_name"] == "cloud"
    assert client.calls[0]["role"] == "synthesizer"


@pytest.mark.asyncio
async def test_router_chat_model_bind_tools_forwards_to_complete() -> None:
    client = _FakeModelClient()
    deps = _make_deps(client)
    req = _make_request()

    class MySchema(BaseModel):
        answer: str

    proxy = build_router_configurable_model(deps=deps, request=req, run_id=uuid4())
    chain = (
        proxy.bind_tools([MySchema])
        .with_config({"model": "planner"})
    )
    await chain.ainvoke([HumanMessage(content="hi")])

    call = client.calls[0]
    assert "tools" in call and len(call["tools"]) == 1
    spec = call["tools"][0]
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "MySchema"


@pytest.mark.asyncio
async def test_router_chat_model_parses_assistant_tool_calls() -> None:
    raw_tc = [
        SimpleNamespace(
            id="call_42",
            function=SimpleNamespace(name="ConductResearch", arguments='{"research_topic": "X"}'),
        )
    ]
    client = _FakeModelClient(response_text="", tool_calls=raw_tc)
    deps = _make_deps(client)
    req = _make_request()

    proxy = build_router_configurable_model(deps=deps, request=req, run_id=uuid4())
    chain = proxy.with_config({"model": "planner"})
    msg = await chain.ainvoke([HumanMessage(content="hi")])

    assert len(msg.tool_calls) == 1
    tc = msg.tool_calls[0]
    assert tc["name"] == "ConductResearch"
    assert tc["args"] == {"research_topic": "X"}
    assert tc["id"] == "call_42"


@pytest.mark.asyncio
async def test_router_configurable_model_queues_declarative_ops_in_order() -> None:
    client = _FakeModelClient()
    deps = _make_deps(client)
    req = _make_request()

    proxy = build_router_configurable_model(deps=deps, request=req, run_id=uuid4())

    # bind_tools is queued by our __getattr__ (it lives on BaseChatModel,
    # not Runnable). `with_retry` is inherited from Runnable, which
    # wraps the proxy in a RunnableRetry rather than calling back into
    # our queue — that's fine, the wrapping preserves correctness.
    p1 = proxy.bind_tools([])
    assert isinstance(p1, RouterConfigurableModel)
    op_names = [op[0] for op in p1._queued_ops]
    assert "bind_tools" in op_names

    # End-to-end the chain still works: bind_tools -> with_retry ->
    # with_config -> ainvoke materializes via our proxy.
    chain = p1.with_retry(stop_after_attempt=2).with_config({"model": "planner"})
    await chain.ainvoke([HumanMessage(content="hi")])
    assert client.calls[-1]["role"] == "planner"


@pytest.mark.asyncio
async def test_router_configurable_model_errors_without_role() -> None:
    client = _FakeModelClient()
    deps = _make_deps(client)
    req = _make_request()

    proxy = build_router_configurable_model(deps=deps, request=req, run_id=uuid4())
    with pytest.raises(RuntimeError, match="role"):
        await proxy.ainvoke([HumanMessage(content="hi")])


@pytest.mark.asyncio
async def test_router_chat_model_converts_message_types() -> None:
    client = _FakeModelClient()
    deps = _make_deps(client)
    req = _make_request()

    proxy = build_router_configurable_model(deps=deps, request=req, run_id=uuid4())
    chain = proxy.with_config({"model": "planner"})
    await chain.ainvoke(
        [
            SystemMessage(content="be terse"),
            HumanMessage(content="ping"),
        ]
    )

    msgs = client.calls[0]["messages"]
    assert msgs[0] == {"role": "system", "content": "be terse"}
    assert msgs[1] == {"role": "user", "content": "ping"}
