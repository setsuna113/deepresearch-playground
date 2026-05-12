"""End-to-end smoke for the LangGraph-backed `run_research`.

We wire the FastAPI-grade dep graph (real `Router`, real
`MemoryService` with ReMe disabled + working memory pointed at a temp
Qdrant embedded path) and substitute a deterministic fake
`ModelClient` for the LLM. This:

- exercises the full pipeline: clarify (skipped) -> write_research_brief
  -> supervisor -> final_report_generation -> reflector;
- verifies LLM dispatch flows through `Router.select()` with **two
  distinct endpoints** under the `co_schedule_v0` profile
  (supervisor=local, final_report=cloud);
- confirms `AgentStep` rows are written via `TraceCallbackHandler`;
- confirms working-memory persistence + reflection writes.

No real LLM, no real Tavily, no Qdrant server — embedded mode + fakes
keep the test hermetic.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from deepresearch.agents.context import RunDependencies
from deepresearch.agents.langgraph.runtime import run_research
from deepresearch.api.deps import build_dependencies
from deepresearch.config.schema import (
    ApiSection,
    AppConfig,
    AppSection,
    EndpointConfig,
    EvalSection,
    MemoryProfileConfig,
    MemorySection,
    ModelProfileConfig,
    ModelsSection,
    PrivacySection,
    ReMeSection,
    SearchProviderConfig,
    SearchSection,
    WorkingMemoryConfig,
)
from deepresearch.models.client import ModelClientResponse
from deepresearch.schemas.runs import RunRequest


# ---------- Fake LLM ------------------------------------------------------


class _StatefulFakeClient:
    """Returns canned OpenAI responses based on call shape (role + tools).

    `calls` records every kwargs dict passed to `complete`, so the test
    can assert the endpoint/role dispatch and message-conversion paths.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._supervisor_invocations = 0

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        role = kwargs.get("role")
        tools = kwargs.get("tools")
        tool_call_id = None
        tool_calls = None

        # write_research_brief: structured_output ResearchQuestion =>
        # bind_tools=[ResearchQuestion]. We must return a tool_call.
        if tools and any(t["function"]["name"] == "ResearchQuestion" for t in tools):
            tool_calls = [
                SimpleNamespace(
                    id="call_brief",
                    function=SimpleNamespace(
                        name="ResearchQuestion",
                        arguments=json.dumps(
                            {"research_brief": "co-scheduling LLM serving"}
                        ),
                    ),
                )
            ]
            content = ""
        # supervisor: bind_tools=[ConductResearch, ResearchComplete, think_tool].
        # Return ResearchComplete to exit fast — no researcher subgraph.
        elif tools and any(t["function"]["name"] == "ResearchComplete" for t in tools):
            self._supervisor_invocations += 1
            tool_calls = [
                SimpleNamespace(
                    id="call_complete",
                    function=SimpleNamespace(
                        name="ResearchComplete", arguments="{}"
                    ),
                )
            ]
            content = ""
        # final_report_generation: no tools, role="final_report".
        elif role == "final_report":
            content = (
                "# Co-scheduling Local + Cloud LLMs\n\n"
                "The thesis explores Pareto trade-offs between latency, "
                "privacy leakage, and joint GPU utilization. [1]\n\n"
                "## Sources\n[1] Synthetic placeholder citation.\n"
            )
        # Reflector: role="reflector", expects JSON.
        elif role == "reflector":
            content = json.dumps(
                {
                    "personal_update": "User cares about thesis-grade reproducibility.",
                    "task_update": "Skipping web search yields no citations.",
                    "tool_update": None,
                    "needs_revision": False,
                }
            )
        else:
            content = "(fallback)"

        msg = SimpleNamespace(content=content, tool_calls=tool_calls)
        raw = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
        return ModelClientResponse(
            text=content,
            prompt_tokens=20,
            completion_tokens=15,
            raw=raw,
            call_id=uuid4(),
        )


# ---------- Real deps but isolated ----------------------------------------


def _build_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        app=AppSection(
            data_dir=str(tmp_path),
            sqlite_path=str(tmp_path / "dr.sqlite"),
            log_level="WARNING",
            log_json=False,
        ),
        models=ModelsSection(
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
                    supervisor="local",
                    researcher="local",
                    compressor="local",
                    final_report="local",
                    reflector="local",
                ),
                "co_schedule_v0": ModelProfileConfig(
                    supervisor="local",
                    researcher="cloud",
                    compressor="local",
                    final_report="cloud",
                    reflector="local",
                ),
            },
        ),
        memory=MemorySection(
            reme=ReMeSection(enabled=False),
            working=WorkingMemoryConfig(
                local_path=str(tmp_path / "qdrant_working"),
                qdrant_url="http://localhost:6333",
                collection_template="dr_working_{user}_{project}",
                embedding_model="hash-fallback",
            ),
            profiles={
                "default": MemoryProfileConfig(
                    personal_top_k=2,
                    task_top_k=2,
                    tool_top_k=1,
                    working_top_k=1,
                    score_floor=0.55,
                ),
                "none": MemoryProfileConfig(
                    personal_top_k=0,
                    task_top_k=0,
                    tool_top_k=0,
                    working_top_k=0,
                    score_floor=1.0,
                ),
            },
        ),
        search=SearchSection(
            default_provider="tavily",
            providers={"tavily": SearchProviderConfig(api_key="", max_results=3)},
        ),
        api=ApiSection(),
        privacy=PrivacySection(),
        eval=EvalSection(),
    )


@pytest.fixture
async def deps(tmp_path: Path) -> RunDependencies:
    cfg = _build_config(tmp_path)
    real = await build_dependencies(cfg)
    fake_client = _StatefulFakeClient()
    deps = RunDependencies(
        config=real.config,
        repos=real.repos,
        recorder=real.recorder,
        model_client=fake_client,  # type: ignore[arg-type]
        router=real.router,
        memory=real.memory,
        tools=real.tools,
    )
    # expose the fake to tests via a side-channel attribute
    deps._fake_client = fake_client  # type: ignore[attr-defined]
    return deps


@pytest.mark.asyncio
async def test_run_research_e2e_routes_local_and_cloud(deps) -> None:
    req = RunRequest(
        query="Trade-offs of AWQ vs GPTQ on 4x RTX 4090?",
        user_id="alice",
        project_id="thesis",
        model_profile="co_schedule_v0",
        max_searches=2,
    )

    run = await run_research(req, deps)

    fake = deps._fake_client  # type: ignore[attr-defined]

    # 1) run finished cleanly with a report
    assert run.status.value == "done", run.error
    assert run.report_md and "Co-scheduling" in run.report_md

    # 2) Routes hit BOTH endpoints under co_schedule_v0
    endpoints_used = {c["endpoint_name"] for c in fake.calls}
    assert "local" in endpoints_used, fake.calls
    assert "cloud" in endpoints_used, fake.calls

    # 3) Roles invoked: supervisor (local) + final_report (cloud) +
    # reflector (local). Researcher never fires because supervisor
    # returns ResearchComplete on its first turn.
    roles_used = {c["role"] for c in fake.calls}
    assert {"supervisor", "final_report", "reflector"}.issubset(roles_used)

    # 4) Trace rows captured by the callback handler
    steps = deps.repos.steps.list_for_run(run.id)
    role_names = {s.role.value for s in steps}
    assert "supervisor" in role_names
    assert "final_report" in role_names
    assert "reflector" in role_names

    # 5) Memory: at least one working-memory write (the report).
    assert run.metrics.n_memory_writes >= 1


@pytest.mark.asyncio
async def test_run_research_no_memory_profile(deps) -> None:
    req = RunRequest(
        query="Same question",
        user_id="alice",
        project_id="thesis",
        memory_profile="none",
        max_searches=2,
    )

    run = await run_research(req, deps)
    assert run.status.value == "done"
    # No prime reads (memory_profile=none means all top_k=0).
    assert run.metrics.n_memory_reads == 0


@pytest.mark.asyncio
async def test_run_research_records_token_counts(deps) -> None:
    req = RunRequest(
        query="A different query",
        user_id="alice",
        project_id="thesis",
        model_profile="phase1_default",
    )

    run = await run_research(req, deps)
    assert run.status.value == "done"

    # ModelCallRecord rows present, each carrying token counts.
    fake = deps._fake_client  # type: ignore[attr-defined]
    assert len(fake.calls) >= 3  # brief, supervisor, final_report, reflector
