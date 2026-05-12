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

### Patch 1: import-prefix rewrite (mechanical)
- Files affected: `deep_researcher.py`
- Change: `from open_deep_research.X import Y` →
  `from deepresearch.agents.langgraph.upstream.X import Y`.
- Reason: vendored package path differs from upstream's installable
  package path. Mechanical; no semantic change.

### Patch 2: model factory hook (planned, commit 3)
- Files affected: `deep_researcher.py`
- Change: replace the module-level
  `configurable_model = init_chat_model(configurable_fields=("model", "max_tokens", "api_key"))`
  with a constructor that resolves to our `RouterChatModel` (via a
  factory passed in `RunnableConfig.configurable`).
- Reason: every LLM call must dispatch through our
  `models.router.Router.select()` — the Phase-4 ParetoDispatch seam.
- Status: applied in commit 3 of this PR.

### Patch 3: reflector node (planned, commit 3)
- Files affected: `deep_researcher.py`
- Change: add a `reflector` node after `final_report_generation` so we
  can emit a `ReflectionUpdate` and trigger memory writes.
- Status: applied in commit 3 of this PR.

## Re-sync procedure

If we need to pull a newer upstream:

1. `gh api repos/langchain-ai/open_deep_research/branches/main --jq '.commit.sha'`
2. Re-fetch the five files at the new SHA.
3. Re-apply the patch list above to the new versions.
4. Update the pinned commit SHA at the top of this file.
5. Re-run the bundled smoke gate.
