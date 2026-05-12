"""End-to-end demo of the LangGraph deep-research pipeline.

Runs a single research query through the full pipeline with FAKE LLM
endpoints (no real vLLM, no Tavily, no ReMe). The fake `ModelClient`
returns canned OpenAI-style responses keyed by the role our `Router`
selected, so we can clearly observe:

- Which role -> which endpoint (local vs. cloud) the router chose.
- The token counts each fake "endpoint" produced.
- The agent step sequence emitted by `TraceCallbackHandler` into SQLite.
- The final report and reflection.

Storage uses a temp SQLite + embedded Qdrant; no external services
required. Run with:

    uv run --extra dev python scripts/demo_e2e.py

For a more thorough view, add `--profile co_schedule_v0` to route
researcher + final_report to "cloud" (the fake still uses different
endpoint labels so you can see both lights flicker).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import tempfile

# Quiet the noisy deprecation warnings + transformers banner before any
# imports that trigger them.
import warnings
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from deepresearch.agents.context import RunDependencies  # noqa: E402
from deepresearch.agents.langgraph.runtime import run_research  # noqa: E402
from deepresearch.api.deps import build_dependencies  # noqa: E402
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
from deepresearch.schemas.runs import RunRequest  # noqa: E402

# ----------------------------------------------------------------------
# Fake LLM that produces tokens.
# ----------------------------------------------------------------------


class FakeRoutingClient:
    """Returns OpenAI-style responses based on (role, tools) shape.

    Each invocation logs: endpoint hit, role, prompt+completion tokens,
    so the operator can SEE which side of the local/cloud split fired.
    """

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
            # write_research_brief — structured output ResearchQuestion.
            content = ""
            tool_calls = [
                SimpleNamespace(
                    id="brief_1",
                    function=SimpleNamespace(
                        name="ResearchQuestion",
                        arguments=json.dumps(
                            {
                                "research_brief": (
                                    "Compare AWQ vs GPTQ on 70B inference: "
                                    "quality, latency, memory, ecosystem support."
                                )
                            }
                        ),
                    ),
                )
            ]
        elif tools and any(t["function"]["name"] == "ResearchComplete" for t in tools):
            # supervisor — return ResearchComplete to skip researcher subgraph
            # for this hermetic demo.
            content = ""
            tool_calls = [
                SimpleNamespace(
                    id="done_1",
                    function=SimpleNamespace(name="ResearchComplete", arguments="{}"),
                )
            ]
        elif role == "final_report":
            content = (
                "# AWQ vs. GPTQ on 4x RTX 4090 — Trade-offs\n\n"
                "**Quality.** Both quantizations preserve ≥98% of FP16 quality on "
                "common benchmarks for 70B-class models; AWQ tends to win on "
                "instruction-following while GPTQ wins on math-heavy prompts. [1]\n\n"
                "**Latency & memory.** AWQ kernels in vLLM achieve roughly 1.2x the "
                "tokens/sec of GPTQ at the same batch size on Ada-class GPUs, with "
                "comparable VRAM (~38 GB for a 70B model on 4x 4090). [2]\n\n"
                "**Ecosystem.** AWQ is first-class in vLLM and SGLang; GPTQ has "
                "broader text-generation-webui support but lags on production "
                "serving stacks. [3]\n\n"
                "## Sources\n"
                "[1] Synthetic placeholder citation #1.\n"
                "[2] Synthetic placeholder citation #2.\n"
                "[3] Synthetic placeholder citation #3.\n"
            )
        elif role == "reflector":
            content = json.dumps(
                {
                    "personal_update": "User cares about thesis-grade reproducibility and Pareto trade-offs.",
                    "task_update": "Quantization comparison tasks benefit from structured quality/latency/memory/ecosystem sections.",
                    "tool_update": None,
                    "needs_revision": False,
                }
            )
        else:
            content = "(fallback response)"

        prompt_tokens = 50 + len(json.dumps(kwargs.get("messages", []))) // 80
        completion_tokens = max(8, len(content) // 5)

        self.log.info(
            "llm.call",
            extra={"endpoint": endpoint, "role": role, "prompt_tok": prompt_tokens, "completion_tok": completion_tokens},
        )
        print(
            f"  [llm] endpoint={endpoint:<6} role={role:<13} "
            f"prompt={prompt_tokens:>4}  completion={completion_tokens:>4}  "
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


# ----------------------------------------------------------------------
# Demo config — two endpoints with distinct labels.
# ----------------------------------------------------------------------


def build_demo_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        app=AppSection(
            data_dir=str(tmp_path),
            sqlite_path=str(tmp_path / "demo.sqlite"),
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


async def run_demo(*, profile: str, query: str) -> int:
    log = logging.getLogger("dr.demo")
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")
    print(f"\n{'='*72}\n  DEEP-RESEARCH DEMO  —  profile = {profile}\n{'='*72}")
    print(f"  Query: {query}\n")

    with tempfile.TemporaryDirectory(prefix="dr_demo_") as td:
        tmp_path = Path(td)
        cfg = build_demo_config(tmp_path)
        real = await build_dependencies(cfg)
        fake = FakeRoutingClient(log=log)
        deps = RunDependencies(
            config=real.config,
            repos=real.repos,
            recorder=real.recorder,
            model_client=fake,  # type: ignore[arg-type]
            router=real.router,
            memory=real.memory,
            tools=real.tools,
        )

        req = RunRequest(
            query=query,
            user_id="demo-user",
            project_id="demo-project",
            model_profile=profile,
            max_searches=2,
        )

        print(f"  Routing table for profile '{profile}':")
        for role, ep in cfg.models.profiles[profile].model_dump().items():
            if ep is None:
                continue
            ep_info = cfg.models.endpoints.get(ep)
            label = ep_info.model_id if ep_info else "?"
            print(f"    {role:<13} -> {ep:<6}  ({label})")
        print()
        print("  Running graph (printing each LLM call as it fires):")
        print()

        run = await run_research(req, deps)

        print()
        print(f"  Status: {run.status.value}")
        print(f"  Latency: {run.metrics.total_latency_ms} ms")
        print(f"  Memory reads: {run.metrics.n_memory_reads}, writes: {run.metrics.n_memory_writes}")
        print(f"  Endpoints hit: {sorted({c['endpoint_name'] for c in fake.calls})}")
        print(f"  Roles invoked: {sorted({c['role'] for c in fake.calls})}")
        print()

        steps = deps.repos.steps.list_for_run(run.id)
        print("  AgentStep trace (recorded by TraceCallbackHandler):")
        for s in sorted(steps, key=lambda s: s.seq):
            tag = "ok" if s.status.value == "ok" else s.status.value.upper()
            print(
                f"    seq={s.seq:>2}  role={s.role.value:<13}  "
                f"status={tag:<6}  latency={s.latency_ms:>4}ms  "
                f"parent_seq={s.input.get('parent_seq')!s}"
            )

        if run.report_md:
            print()
            print("  Final report (truncated):")
            for line in run.report_md.splitlines()[:8]:
                print(f"    | {line}")
            if len(run.report_md.splitlines()) > 8:
                print(f"    | ... ({len(run.report_md.splitlines()) - 8} more lines)")

        ok = (
            run.status.value == "done"
            and "local" in {c["endpoint_name"] for c in fake.calls}
            and (profile != "co_schedule_v0" or "cloud" in {c["endpoint_name"] for c in fake.calls})
            and bool(run.report_md)
        )
        print()
        print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--profile",
        default="co_schedule_v0",
        choices=("phase1_default", "co_schedule_v0"),
        help="Which model profile to demo (co_schedule_v0 routes researcher+final_report to cloud).",
    )
    p.add_argument(
        "--query",
        default="What are the trade-offs of AWQ vs GPTQ for 70B inference on 4x RTX 4090?",
    )
    args = p.parse_args()
    return asyncio.run(run_demo(profile=args.profile, query=args.query))


if __name__ == "__main__":
    sys.exit(main())
