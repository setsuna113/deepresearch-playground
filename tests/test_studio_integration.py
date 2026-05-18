"""Studio integration tests.

Covers the Studio variant of the LangGraph pipeline that doesn't run
through `runtime.run_research`:

1. `studio_bootstrap` primes memory, populates the `_studio_active_run`
   slot, and emits a bootstrap `AgentStep` to the trace.
2. `reflector_writer_node` persists `write_reflection` +
   `write_working_report` and finalizes the run row.
3. `agents.langgraph.state.AgentState` declares the `reflection` key —
   regression for the latent reducer-drop bug fixed in the Phase 1.5
   sweep.

These tests mock the bare LLM-touching parts (`reflector_node`,
`_get_deps`) but exercise the real `MemoryService` + `Repositories` so
the wiring around the seam is real.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deepresearch.agents.context import RunDependencies
from deepresearch.agents.langgraph import router_chat_model as _rcm
from deepresearch.agents.langgraph import studio as studio_module
from deepresearch.agents.langgraph.state import AgentState
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
from deepresearch.schemas.agents import AgentRole, ReflectionUpdate, StepStatus
from deepresearch.schemas.runs import RunRequest


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
                    model_id="test-model",
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
    return await build_dependencies(cfg)


@pytest.fixture(autouse=True)
def _reset_studio_slots():
    """Clear module-level caches between tests."""
    studio_module._DEPS_CACHE = None
    _rcm.clear_studio_active_run()
    yield
    studio_module._DEPS_CACHE = None
    _rcm.clear_studio_active_run()


@pytest.mark.asyncio
async def test_studio_bootstrap_primes_and_emits_step(
    deps: RunDependencies, monkeypatch
) -> None:
    """studio_bootstrap should:
    - skip the global deps init (use injected),
    - set the _studio_active_run slot with primes + callback,
    - record a bootstrap AgentStep to the recorder."""

    async def fake_get_deps() -> RunDependencies:
        return deps

    monkeypatch.setattr(studio_module, "_get_deps", fake_get_deps)

    state = {"messages": [{"type": "human", "content": "test query for studio"}]}
    config = {
        "configurable": {
            "user_id": "alice",
            "project_id": "thesis",
            "model_profile": "phase1_default",
            "memory_profile": "default",
        },
        "callbacks": [],
    }

    result = await studio_module.studio_bootstrap(state, config)  # type: ignore[arg-type]
    assert result == {}

    # Active run slot populated with all the runtime attributes.
    active = _rcm._studio_active_run
    assert active is not None
    assert active.request.query == "test query for studio"
    assert active.request.user_id == "alice"
    assert getattr(active, "callback_handler", None) is not None
    assert getattr(active, "seq_alloc", None) is not None
    assert getattr(active, "memory_profile", None) is not None

    # Callback attached to the config so downstream nodes' chain events
    # land on our TraceCallbackHandler.
    assert config["callbacks"], "TraceCallbackHandler should be appended to config[callbacks]"

    # A bootstrap step was recorded to SQLite.
    steps = deps.repos.steps.list_for_run(active.run_id)
    bootstrap_steps = [s for s in steps if s.role == AgentRole.planner]
    assert len(bootstrap_steps) == 1
    bs = bootstrap_steps[0]
    assert bs.status == StepStatus.ok
    assert bs.input.get("node") == "studio_bootstrap"
    assert bs.output.get("model_profile") == "phase1_default"


@pytest.mark.asyncio
async def test_studio_bootstrap_emits_error_step_when_prime_fails(
    deps: RunDependencies, monkeypatch
) -> None:
    """If prime_brief_messages raises, the bootstrap step should still
    be recorded but with status=error and a prime_error in output."""

    async def fake_get_deps() -> RunDependencies:
        return deps

    async def boom(**_kwargs):
        raise RuntimeError("memory layer offline")

    monkeypatch.setattr(studio_module, "_get_deps", fake_get_deps)
    monkeypatch.setattr(studio_module, "prime_brief_messages", boom)

    state = {"messages": [{"type": "human", "content": "q"}]}
    config: dict = {"configurable": {}, "callbacks": []}

    await studio_module.studio_bootstrap(state, config)  # type: ignore[arg-type]
    active = _rcm._studio_active_run
    assert active is not None

    steps = deps.repos.steps.list_for_run(active.run_id)
    bootstrap_steps = [s for s in steps if s.role == AgentRole.planner]
    assert len(bootstrap_steps) == 1
    bs = bootstrap_steps[0]
    assert bs.status == StepStatus.error
    assert bs.error == "RuntimeError"
    assert bs.output.get("prime_error") == "RuntimeError"


@pytest.mark.asyncio
async def test_reflector_writer_persists_reflection_and_report(
    deps: RunDependencies, monkeypatch
) -> None:
    """reflector_writer_node calls write_reflection + write_working_report
    and marks the run done."""

    from uuid import uuid4 as _uuid4

    req = RunRequest(
        query="reflector test",
        user_id="alice",
        project_id="thesis",
        memory_profile="default",
    )
    run_id = _uuid4()

    # Pre-seed an active run row so mark_done has something to update.
    from deepresearch.schemas.runs import ResearchRun, RunStatus
    deps.repos.runs.create(ResearchRun(id=run_id, request=req, status=RunStatus.running))

    # Manually populate the active-run slot with the runtime attributes
    # studio_bootstrap normally sets.
    from deepresearch.agents.langgraph.callbacks import SeqAllocator, TraceCallbackHandler
    seq_alloc = SeqAllocator()
    active = _rcm._ActiveRun(deps=deps, request=req, run_id=run_id)
    active.callback_handler = TraceCallbackHandler(  # type: ignore[attr-defined]
        recorder=deps.recorder, run_id=run_id, seq_alloc=seq_alloc
    )
    active.seq_alloc = seq_alloc  # type: ignore[attr-defined]
    active.t0 = 0.0  # type: ignore[attr-defined]
    _rcm.set_studio_active_run(active)

    # Mock reflector_node to return a known reflection dict.
    async def fake_reflector_node(state, config):
        return {
            "reflection": {
                "personal_update": "User cares about reproducibility.",
                "task_update": "Use minimal preset.",
                "tool_update": None,
                "needs_revision": False,
            }
        }

    monkeypatch.setattr(studio_module, "reflector_node", fake_reflector_node)

    # Spies for the persistence helpers.
    captured: dict = {}

    async def fake_write_reflection(*, deps, run_id, request, reflection, **kw):
        captured["reflection"] = reflection
        return 2  # n_written

    async def fake_write_working_report(*, deps, run_id, request, report, **kw):
        captured["report"] = report
        return object()

    monkeypatch.setattr(studio_module, "write_reflection", fake_write_reflection)
    monkeypatch.setattr(studio_module, "write_working_report", fake_write_working_report)

    state = {
        "messages": [],
        "final_report": "# Final\n\nA tidy report. [1]",
        "reflection": None,
    }

    result = await studio_module.reflector_writer_node(state, {})  # type: ignore[arg-type]

    # Reflection dict propagates back into state.
    assert result.get("reflection") is not None

    # Both writes fired with the expected payloads.
    assert isinstance(captured.get("reflection"), ReflectionUpdate)
    assert captured["reflection"].personal_update.startswith("User cares")
    assert captured.get("report") == "# Final\n\nA tidy report. [1]"

    # A finalize marker step was recorded with role=reflector, status=ok.
    steps = deps.repos.steps.list_for_run(run_id)
    reflector_steps = [s for s in steps if s.role == AgentRole.reflector]
    assert len(reflector_steps) == 1
    rs = reflector_steps[0]
    assert rs.status == StepStatus.ok
    assert rs.input.get("node") == "reflector_writer"
    assert rs.output.get("n_writes") == 3  # 2 reflection + 1 report

    # Run row marked done.
    got = deps.repos.runs.get(run_id)
    assert got is not None
    assert got.status == RunStatus.done


def test_agent_state_declares_reflection_key() -> None:
    """Regression for the latent bug where the upstream AgentState
    TypedDict didn't list `reflection`, so LangGraph's reducer silently
    dropped the reflector's output. Our state.py subclass must declare
    it."""
    annotations = getattr(AgentState, "__annotations__", {})
    assert "reflection" in annotations, (
        "agents.langgraph.state.AgentState must declare a 'reflection' "
        "field or the LangGraph reducer will drop the reflector's output"
    )
