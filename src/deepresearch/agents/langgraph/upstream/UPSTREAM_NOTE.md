# Vendored: open_deep_research

**Source:** https://github.com/langchain-ai/open_deep_research
**Pinned commit:** `0dd30bd47ed6ed3ac4d2b678997662830f227a14` (2026-04-28)
**License:** MIT (Copyright (c) 2025 LangChain) — see LICENSE-UPSTREAM.txt

## Files

All five files come from `src/open_deep_research/` in the source repo:

- `deep_researcher.py` — top-level graph + supervisor subgraph + researcher subgraph
- `state.py` — TypedDicts (`AgentState`, `SupervisorState`, `ResearcherState`)
- `configuration.py` — `Configuration` dataclass + `SearchAPI` enum
- `utils.py` — `get_all_tools`, search adapters, MCP loading
- `prompts.py` — system prompts

## Imports

Upstream imports `from open_deep_research.{...}`. We import as
`from deepresearch.agents.langgraph.upstream.{...}`. Two ways this is
handled:

1. Within these vendored files, imports are **rewritten** to use the
   `deepresearch.agents.langgraph.upstream.*` prefix.
2. Our integration code (`agents.langgraph.runtime`,
   `agents.langgraph.router_chat_model`, etc.) imports these files
   directly through their vendored path.

## Patches applied

Each item below records one local edit on top of the pinned source.
Numbers reflect creation order, not file order — Patch 5 predates
Patch 4, and Patches 6/7 were added together after the Patch-5
follow-on review.

### Patch 1: import-prefix rewrite (mechanical)
- Files affected: `deep_researcher.py`, `utils.py`
- Change: `from open_deep_research.X import Y` →
  `from deepresearch.agents.langgraph.upstream.X import Y`.
- Reason: vendored package path differs from upstream's installable
  package path. Mechanical; no semantic change.

### Patch 2: model factory hook
- Files affected: `deep_researcher.py`
- Change: replaced the module-level
  `configurable_model = init_chat_model(configurable_fields=("model", "max_tokens", "api_key"))`
  with `from deepresearch.agents.langgraph.router_chat_model import configurable_model_proxy as configurable_model`.
  The proxy resolves to a per-run `RouterConfigurableModel` via a
  contextvar bound by `runtime.run_research`.
- Reason: every LLM call must dispatch through our
  `models.router.Router.select()` — the Phase-4 ParetoDispatch seam.

### Patch 3: reflector node — additive, in runtime not in vendored code
- Files affected: NONE — the reflector node lives in
  `deepresearch.agents.langgraph.reflection_node` and is added to a
  *new* top-level `StateGraph` built by `runtime._build_graph()`. The
  vendored `deep_researcher_builder` / `deep_researcher` symbols are
  no longer used by us, but remain in the file so the upstream public
  surface is intact for future re-sync.

### Patch 5: defensive `response is None` fallback in `write_research_brief`
- Files affected: `deep_researcher.py`
- Change: when `research_model.ainvoke(...)` returns None (structured
  output parser couldn't extract a `ResearchQuestion` after
  `max_structured_output_retries`), fall back to the raw last user
  message as the research brief instead of raising
  `AttributeError("'NoneType' object has no attribute 'research_brief'")`.
- Reason: smaller local models (Qwen3-8B-AWQ on our laptop, similar
  parameter-class quantized models) fumble multi-turn structured
  output with low probability per attempt; over the lifetime of a
  thesis evaluation pass this would otherwise kill many runs at
  step 2 of 7. The fallback keeps the pipeline running with a
  degraded brief instead of an empty SQLite row.

### Patch 4: summarization model factory hook in `utils.py`
- Files affected: `utils.py`
- Change: replaced `summarization_model = init_chat_model(...)` inside
  `tavily_search` with a call to
  `deepresearch.agents.langgraph.router_chat_model.get_active_router_model()`.
- Reason: keep Tavily's per-page summarization on the Router seam too.
  This path only fires when `search_api == TAVILY`; the Phase-1.5
  smoke gate runs with `search_api == NONE` so this patch is dormant.

### Patch 6: propagate non-token-limit exceptions in `supervisor_tools`
- Files affected: `deep_researcher.py`
- Change: in the `except Exception as e:` block of `supervisor_tools`,
  removed the unconditional `or True` from
  `if is_token_limit_exceeded(e, configurable.research_model) or True:`
  and added `raise` after the token-limit branch. The token-limit case
  still exits gracefully to END; every other exception bubbles up.
- Reason: upstream swallowed every failure to `goto=END` regardless of
  cause, which hid transient ReMe/SQLite/network errors as if they were
  token-limit overflows. For thesis-grade metrics we want the trace to
  show the real failure: non-token-limit exceptions propagate to
  `runtime.run_research`, which logs `run_failed` and persists the error
  on the `ResearchRun.error` column.

### Patch 7: defensive None/empty response fallback in supervisor + researcher
- Files affected: `deep_researcher.py`
- Change: after `response = await research_model.ainvoke(...)` in both
  the `supervisor` node and the `researcher` node, check whether
  `response` is None or carries neither tool_calls nor non-empty content.
  If so, replace it with an `AIMessage` whose only tool call is a
  synthesized `ResearchComplete` (with id `synth_research_complete` so
  the synthetic origin is visible in the trace).
- Reason: small local models (Qwen3-8B-AWQ class, similar) occasionally
  exhaust `max_structured_output_retries` and return None or an empty
  AIMessage. Downstream code reads `most_recent_message.tool_calls`; a
  None response would crash with `AttributeError`, and an empty response
  in the researcher would silently loop until `MAX_REACT_TOOL_CALLS`.
  Synthesizing a deterministic `ResearchComplete` makes the failure
  mode appear as an intentional, visible exit instead.

## Re-sync procedure

If we need to pull a newer upstream:

1. `gh api repos/langchain-ai/open_deep_research/branches/main --jq '.commit.sha'`
2. Re-fetch the five files at the new SHA.
3. Re-apply the patch list above to the new versions.
4. Update the pinned commit SHA at the top of this file.
5. Re-run the bundled smoke gate.
