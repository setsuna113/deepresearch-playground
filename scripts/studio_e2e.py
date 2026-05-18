"""End-to-end harness for the LangGraph **Studio** path.

Mirrors `scripts/demo_e2e.py` but invokes the **compiled `studio_graph`**
that `langgraph dev` serves, so the same code path the browser UI
exercises can be driven from the terminal:

    # Hermetic, fake LLM, temp SQLite + embedded Qdrant
    uv run --extra dev python scripts/studio_e2e.py --fake

    # Live: uses config.local.yaml endpoints (vLLM / DeepSeek / etc.)
    uv run --extra dev python scripts/studio_e2e.py \\
        --query "AWQ vs GPTQ for 70B on 4x4090?" \\
        --profile co_schedule_v0

Why this exists:

- The previous-only debug surface for Studio was opening the browser
  tab, which made every iteration cost a paste+wait round trip with
  the operator. With this harness we can stress the same graph the UI
  uses, capture the post-run AgentStep + ModelCall tables in the
  terminal, and iterate in seconds.
- `scripts/demo_e2e.py` drives `runtime.run_research` which is the CLI
  path; it does NOT exercise `studio_bootstrap`, the
  `reflector_writer_node`, or the module-level Studio active-run slot.
  This harness does.

Exit code is 0 on success and 1 on any failure or assertion mismatch.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from langchain_core.messages import HumanMessage  # noqa: E402

from deepresearch.agents.context import RunDependencies  # noqa: E402
from deepresearch.api.deps import build_dependencies  # noqa: E402
from deepresearch.config.loader import get_config  # noqa: E402
from deepresearch.config.schema import (  # noqa: E402
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
from deepresearch.models.client import ModelClientResponse  # noqa: E402

# ----------------------------------------------------------------------
# Fake client (same shape as demo_e2e.FakeRoutingClient, kept here so
# this script is self-contained).
# ----------------------------------------------------------------------


class FakeRoutingClient:
    """Returns OpenAI-style responses based on (role, tools) shape."""

    def __init__(self, log: logging.Logger) -> None:
        self.log = log
        self.calls: list[dict] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        endpoint = kwargs.get("endpoint_name", "?")
        role = kwargs.get("role", "?")
        tools = kwargs.get("tools")

        tool_calls = None
        if tools and any(t["function"]["name"] == "ResearchQuestion" for t in tools):
            content = ""
            tool_calls = [
                SimpleNamespace(
                    id="brief_1",
                    function=SimpleNamespace(
                        name="ResearchQuestion",
                        arguments=json.dumps(
                            {
                                "research_brief": (
                                    "Compare AWQ vs GPTQ on 70B inference on 4x4090: "
                                    "quality, latency, memory, ecosystem."
                                )
                            }
                        ),
                    ),
                )
            ]
        elif tools and any(t["function"]["name"] == "ResearchComplete" for t in tools):
            content = ""
            tool_calls = [
                SimpleNamespace(
                    id="done_1",
                    function=SimpleNamespace(name="ResearchComplete", arguments="{}"),
                )
            ]
        elif role == "final_report":
            content = (
                "# AWQ vs. GPTQ on 4x RTX 4090\n\n"
                "**Quality.** Both preserve ~98% of FP16 on common benchmarks. [1]\n\n"
                "**Latency & memory.** AWQ kernels in vLLM run ~1.2x tokens/sec at "
                "comparable VRAM. [2]\n\n"
                "**Ecosystem.** AWQ is first-class in vLLM/SGLang. [3]\n\n"
                "## Sources\n[1] placeholder.\n[2] placeholder.\n[3] placeholder.\n"
            )
        elif role == "reflector":
            content = json.dumps(
                {
                    "personal_update": "User prefers thesis-grade reproducibility.",
                    "task_update": "Quantization comparison benefits from quality/latency/memory/ecosystem axes.",
                    "tool_update": None,
                    "needs_revision": False,
                }
            )
        else:
            content = "(fallback)"

        prompt_tokens = 50 + len(json.dumps(kwargs.get("messages", []))) // 80
        completion_tokens = max(8, len(content) // 5)

        print(
            f"  [llm] endpoint={endpoint:<6} role={role:<13} "
            f"prompt={prompt_tokens:>4} completion={completion_tokens:>4} "
            f"{'<tool_call>' if tool_calls else '<text>'}"
        )

        msg = SimpleNamespace(content=content, tool_calls=tool_calls)
        raw = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
        return ModelClientResponse(
            text=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw=raw,
            call_id=uuid4(),
        )


def _build_fake_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        app=AppSection(
            data_dir=str(tmp_path),
            sqlite_path=str(tmp_path / "studio_e2e.sqlite"),
            log_level="WARNING",
            log_json=False,
        ),
        models=ModelsSection(
            endpoints={
                "local": EndpointConfig(
                    base_url="http://localhost:8001/v1",
                    api_key="EMPTY",
                    model_id="qwen3-8b-awq (laptop 4090)",
                    role_hint="minimizer",
                ),
                "cloud": EndpointConfig(
                    base_url="http://sjtu:8000/v1",
                    api_key="EMPTY",
                    model_id="qwen2.5-72b-instruct-awq (4x 4090)",
                    role_hint="heavy",
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


async def _run(
    *,
    profile: str,
    query: str,
    use_fake: bool,
    user: str,
    project: str,
) -> int:
    log = logging.getLogger("dr.studio_e2e")
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")

    print(f"\n{'=' * 72}\n  STUDIO E2E  —  profile = {profile}  fake = {use_fake}\n{'=' * 72}")
    print(f"  Query: {query}\n")

    # Build deps. Two paths: (a) fake/hermetic with a temp dir + fake
    # client; (b) live using config.local.yaml's endpoints.
    tmp_ctx: tempfile.TemporaryDirectory | None = None
    if use_fake:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="dr_studio_e2e_")
        tmp_path = Path(tmp_ctx.name)
        cfg = _build_fake_config(tmp_path)
        real_deps = await build_dependencies(cfg)
        fake = FakeRoutingClient(log=log)
        deps = RunDependencies(
            config=real_deps.config,
            repos=real_deps.repos,
            recorder=real_deps.recorder,
            model_client=fake,  # type: ignore[arg-type]
            router=real_deps.router,
            memory=real_deps.memory,
            tools=real_deps.tools,
        )
    else:
        cfg = get_config()
        deps = await build_dependencies(cfg)
        fake = None  # type: ignore[assignment]

    # studio.py caches a deps singleton in `_DEPS_CACHE`; pre-populate
    # so studio_bootstrap reuses ours (especially for the fake case).
    import deepresearch.agents.langgraph.studio as studio_mod

    studio_mod._DEPS_CACHE = deps

    # Show routing table.
    print(f"  Routing table for profile '{profile}':")
    profile_cfg = deps.config.models.profiles.get(profile)
    if profile_cfg is not None:
        for role, ep in profile_cfg.model_dump().items():
            if ep is None:
                continue
            ep_info = deps.config.models.endpoints.get(ep)
            label = ep_info.model_id if ep_info else "?"
            print(f"    {role:<13} -> {ep:<6}  ({label})")
    print()
    print("  Running studio_graph (printing each LLM call as it fires):")
    print()

    initial_state = {"messages": [HumanMessage(content=query)]}
    runnable_config: dict[str, Any] = {
        "configurable": {
            "user_id": user,
            "project_id": project,
            "model_profile": profile,
            "memory_profile": "default",
            "max_searches": 2,
            "max_concurrent_units": 1,
        },
        "recursion_limit": 60,
    }

    final_state = await studio_mod.studio_graph.ainvoke(
        initial_state, config=runnable_config
    )

    # The active-run slot has the run_id we minted in studio_bootstrap.
    from deepresearch.agents.langgraph import router_chat_model as _rcm

    active = _rcm._studio_active_run
    if active is None:
        print("  FAIL: no active run after invocation")
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
        return 1
    run_id = active.run_id

    print()
    print("  AgentStep trace (manual studio markers + LangChain chain events):")
    steps = sorted(deps.repos.steps.list_for_run(run_id), key=lambda s: s.seq)
    if not steps:
        print("    (no steps — chain-event propagation may be unavailable)")
    for s in steps:
        tag = "ok" if s.status.value == "ok" else s.status.value.upper()
        print(
            f"    seq={s.seq:>2}  role={s.role.value:<13}  status={tag:<6}  "
            f"latency={s.latency_ms:>4}ms  parent_seq={s.input.get('parent_seq')!s}"
        )

    print()
    print("  ModelCall rows (per-LLM-call audit):")
    calls = sorted(
        deps.repos.model_calls.list_for_run(run_id), key=lambda c: c.started_at
    )
    if not calls:
        print("    (none)")
    for i, c in enumerate(calls, 1):
        print(
            f"    #{i:>2}  endpoint={c.endpoint_name:<6}  role={c.role:<13}  "
            f"prompt={c.prompt_tokens:>4}  completion={c.completion_tokens:>4}  "
            f"latency={c.latency_ms:>4}ms"
        )

    report = (final_state.get("final_report") or "").strip()
    print()
    print(f"  Final report length: {len(report)} chars")
    if report:
        for line in report.splitlines()[:6]:
            print(f"    | {line}")
        if len(report.splitlines()) > 6:
            print(f"    | ... ({len(report.splitlines()) - 6} more lines)")

    reflection = final_state.get("reflection")
    has_reflection = isinstance(reflection, dict) and any(
        reflection.get(k) for k in ("personal_update", "task_update", "tool_update")
    )
    print()
    print(f"  Reflection emitted: {has_reflection}")
    if has_reflection:
        print(
            f"    personal={reflection.get('personal_update')!r}\n"
            f"    task={reflection.get('task_update')!r}\n"
            f"    tool={reflection.get('tool_update')!r}"
        )

    # Pass criteria differ by mode:
    # - fake: graph completes, report exists, endpoint mix matches profile
    # - live: graph completes, report exists, ModelCallRecord rows exist
    ok = bool(report)
    if use_fake and fake is not None:
        endpoints_hit = {c.get("endpoint_name", "?") for c in fake.calls}
        ok = ok and ("local" in endpoints_hit)
        if profile == "co_schedule_v0":
            ok = ok and ("cloud" in endpoints_hit)
        print()
        print(f"  Fake-client endpoints hit: {sorted(endpoints_hit)}")
    else:
        ok = ok and bool(calls)

    print()
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")

    if tmp_ctx is not None:
        tmp_ctx.cleanup()
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--profile",
        default="phase1_default",
        help="Model profile to use (phase1_default, co_schedule_v0, ...).",
    )
    p.add_argument(
        "--query",
        default="What are the trade-offs of AWQ vs GPTQ for 70B inference on 4x RTX 4090?",
    )
    p.add_argument(
        "--fake",
        action="store_true",
        help="Use the bundled fake ModelClient (hermetic; no live endpoints needed).",
    )
    p.add_argument("--user", default="studio_e2e")
    p.add_argument("--project", default="studio_e2e")
    args = p.parse_args()
    return asyncio.run(
        _run(
            profile=args.profile,
            query=args.query,
            use_fake=args.fake,
            user=args.user,
            project=args.project,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
