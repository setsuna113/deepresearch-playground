"""LangGraph-based deep-research orchestration.

Phase 1.5 onwards, `run_research` is implemented as a LangGraph pipeline
vendored from langchain-ai/open_deep_research (see `upstream/`) and
wired to our `Router`, `MemoryService`, and `TraceRecorder`.

Public surface:

- `run_research(req, deps, *, existing_run=None)` — replaces the
  pre-1.5 STORM loop. The legacy import path
  `deepresearch.agents.orchestrator.run_research` continues to work via
  a shim that forwards here.
"""

from __future__ import annotations

# Re-exports defer to runtime to avoid importing LangChain at package
# import time (CLI startup latency).
__all__ = ["run_research"]


def __getattr__(name: str):  # pragma: no cover - thin lazy shim
    if name == "run_research":
        from deepresearch.agents.langgraph.runtime import run_research

        return run_research
    raise AttributeError(name)
