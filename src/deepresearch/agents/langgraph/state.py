"""Our extension of upstream's `AgentState`.

The vendored `open_deep_research` `AgentState` TypedDict declares only the
fields its own nodes write (`messages`, `supervisor_messages`,
`research_brief`, `raw_notes`, `notes`, `final_report`). When our
`reflector_node` writes a `reflection` key, LangGraph's reducer drops it
because the schema doesn't list it.

This shim subclasses upstream's `AgentState` and declares `reflection`
so the value survives into `final_state` for both `runtime.run_research`
and the Studio path's `reflector_writer_node`.

If we ever re-sync upstream and they add a `reflection` field of their
own, delete this file and re-export upstream's `AgentState` directly.
"""

from __future__ import annotations

from typing import Any

from deepresearch.agents.langgraph.upstream.state import AgentState as UpstreamAgentState


class AgentState(UpstreamAgentState):
    """Upstream AgentState + the `reflection` key our reflector writes."""

    reflection: dict[str, Any] | None


__all__ = ["AgentState"]
