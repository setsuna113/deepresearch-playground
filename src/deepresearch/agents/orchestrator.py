"""Orchestrator entrypoint — thin shim over the LangGraph runtime.

Pre-Phase 1.5 this module hosted a hand-rolled STORM loop. Phase 1.5
onwards the agent flow lives in `deepresearch.agents.langgraph.runtime`,
built on a vendored `open_deep_research` graph. We keep this module so
the public symbol path (`deepresearch.agents.orchestrator.run_research`)
stays stable for `cli/run.py`, `api/routes/runs.py`, and
`eval/runner.py`.
"""

from __future__ import annotations

from deepresearch.agents.langgraph.runtime import run_research

__all__ = ["run_research"]
