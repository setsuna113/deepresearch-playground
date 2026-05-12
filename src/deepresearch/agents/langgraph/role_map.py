"""Single source of truth mapping LangGraph node names to our `AgentRole`.

The vendored `open_deep_research` graph has four top-level nodes plus a
supervisor subgraph that spawns researcher subgraphs. We map each node
to a coarse role so that:

- `Router.select(profile, role, envelope, hint)` can pick an endpoint.
- `TraceCallbackHandler` can tag each emitted `AgentStep` with the
  right role for thesis-grade metric attribution.

The mapping is intentionally coarse — clarify / brief / supervisor share
`supervisor` because upstream's `Configuration.research_model` field
drives all three. If thesis evaluation later needs finer-grained
attribution, split the roles here and patch the Configuration consumers.
"""

from __future__ import annotations

from deepresearch.schemas.agents import AgentRole

# LangGraph node name (as registered with StateGraph.add_node) -> role.
NODE_TO_ROLE: dict[str, AgentRole] = {
    "clarify_with_user": AgentRole.supervisor,
    "write_research_brief": AgentRole.supervisor,
    "research_supervisor": AgentRole.supervisor,
    "supervisor": AgentRole.supervisor,
    "supervisor_tools": AgentRole.supervisor,
    "researcher": AgentRole.researcher,
    "researcher_tools": AgentRole.researcher,
    "compress_research": AgentRole.compressor,
    "final_report_generation": AgentRole.final_report,
    "reflector": AgentRole.reflector,
}


# Reverse: which roles are routed by which upstream `Configuration` field.
# Used by `runtime.py` when building the RunnableConfig that drives
# vendored nodes: the field's string value becomes the role string our
# RouterChatModel reads from `.with_config({"model": ...})`.
CONFIG_FIELD_TO_ROLE: dict[str, str] = {
    "research_model": AgentRole.supervisor.value,
    "summarization_model": AgentRole.compressor.value,
    "compression_model": AgentRole.compressor.value,
    "final_report_model": AgentRole.final_report.value,
}


def role_for_node(node_name: str) -> AgentRole | None:
    """Return the `AgentRole` registered for a LangGraph node, or None.

    Returning None (rather than raising) lets the callback handler skip
    unknown internal nodes that LangGraph emits (e.g., `__start__`).
    """
    return NODE_TO_ROLE.get(node_name)
