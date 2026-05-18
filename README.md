# deepresearch-playground

A research playground for studying co-scheduling between a local RTX 4090
(running a ≤20B minimizer/personalizer + private memory + RAG) and a
4×RTX 4090 cloud node (running a 70B AWQ model via vLLM/SGLang), with the
goal of traversing a 3-D Pareto frontier of **latency × privacy-leakage ×
joint-GPU-utilization** for deep-research workflows.

The thesis combines three streams that have not yet been integrated:

1. PD-disaggregation / agentic schedulers (DistServe, Llumnix, Mooncake,
   Parrot, ECO-LLM).
2. Hierarchical memory OSes (MemOS, Letta/MemGPT, A-Mem, Mem0, Zep/Graphiti,
   **ReMe**).
3. Contextual-integrity privacy enforcement for multi-agent traffic
   (AirGapAgent, PrivacyLens, MAGPIE, AgentLeak).

## Novel contributions (phased)

- **ParetoDispatch** — privacy-bounded co-scheduling (Phase 4).
- **Reflection Broadcast Protocol** — CI-filtered reflection propagation
  under a mutual-information budget (Phase 5).
- **Hot/warm/cold hierarchical memory placement** across local-GPU /
  local-SSD / cloud tiers (Phase 5).
- **BrowseComp-Hybrid** — BrowseComp + GAIA + LoCoMo extension (Phase 6).

## Phase 1.5 (current)

The orchestrator is built on a vendored **LangChain `open_deep_research`**
LangGraph pipeline (pinned to commit `0dd30bd` from 2026-04-28, MIT) plus
a small set of our own modules that keep the Phase-4 ParetoDispatch seam
clean:

- **Graph** (`agents/langgraph/upstream/` + `runtime.py`):
  `clarify_with_user → write_research_brief → research_supervisor →
  final_report_generation → reflector → END`. Supervisor + researcher
  ReAct loops come from upstream; the **reflector** is our injected node.
- **Router seam**: every LLM call materializes through
  `RouterChatModel` → `ModelClient.complete()` → `Router.select(profile,
  role, envelope, hint)`. Two `Configuration` model fields drive role
  selection: `research_model` → supervisor/researcher, `final_report_model`
  → final synthesis, `compression_model` → compressor.
- **Memory**: personal / task / tool via ReMe (`reme_ai.ReMeApp` flowllm
  flows), plus a working-memory Qdrant collection we manage ourselves.
- **Tracing**: a `TraceCallbackHandler` maps LangGraph node events to
  `AgentStep` rows in SQLite. Parent/child seq IDs let parallel researcher
  subgraphs land distinct rows.
- **Storage**: SQLite for runs / traces / metrics; embedded Qdrant for
  working memory by default (no Docker needed).
- **Runtime**: FastAPI service + Typer CLI. All LLM calls go through an
  OpenAI-compatible `ModelClient`.

### Hermetic e2e demo (no external services)

```bash
uv run --extra dev python scripts/demo_e2e.py --profile co_schedule_v0
```

Runs the full pipeline against a fake `ModelClient` so you can SEE
which roles route to local vs. cloud endpoints, the token counts each
fake "endpoint" produced, and the AgentStep trace written to SQLite.

## Quickstart

```bash
# 1. install
uv sync

# 2. start Qdrant (Docker)
bash scripts/run_qdrant_local.sh

# 3. configure
cp config/config.example.yaml config/config.local.yaml
cp .env.example .env
# fill in TAVILY_API_KEY and any model endpoints

# 4. run a research query
uv run deepresearch run \
  "Trade-offs of AWQ vs GPTQ for 70B inference on 4xRTX4090?" \
  --user alice --project thesis --depth standard

# 5. serve the API
uv run uvicorn deepresearch.api.app:create_app --factory --port 8765
```

### ReMe embedding endpoint

ReMe summary flows require an OpenAI-compatible `/v1/embeddings` endpoint.
A vLLM server hosting a causal LM (Qwen3-8B-AWQ etc.) and DeepSeek's chat
API both lack this endpoint, so set the embedding endpoint separately in
`.env`:

```bash
REME_EMBEDDING_API_BASE=https://api.openai.com/v1
REME_EMBEDDING_API_KEY=$OPENAI_API_KEY
```

with `memory.reme.embedding.model_id: text-embedding-3-small` in
`config.local.yaml`. Cost is ~$0.02 / 1M tokens. If unset, the adapter
falls back to the LLM endpoint and ReMe writes will fail-soft at the
embedding step (reads still work for memories written via a properly
configured pass).

## Layout

```
src/deepresearch/
  schemas/        # pydantic v2 data shapes
  config/         # YAML + env config loader
  storage/        # SQLite (SQLModel) tables and repos
  models/         # OpenAI-compatible client + Router (Phase-4 seam)
  memory/         # ReMe adapter + working memory
  tools/          # search providers + page fetch
  agents/         # langgraph/{runtime,router_chat_model,callbacks,reflection_node,memory_hooks,upstream/}
  api/            # FastAPI
  cli/            # Typer entry: `deepresearch`
  observability/  # structlog + trace recorder
  eval/           # golden suite runner
  scheduling/     # Phase-4 placeholder (ParetoDispatch)
  privacy/        # Phase-4-5 placeholder (CI envelopes)
  broadcast/      # Phase-5 placeholder (Reflection Broadcast Protocol)
  tiering/        # Phase-5 placeholder (hot/warm/cold placement)
```

## Status

Phase 1.5: LangGraph orchestrator + ReMe wire-up landed (2026-05-12).
See `/home/lyc/.claude/plans/i-made-a-dapper-quasar.md` for the
adoption plan and `src/deepresearch/agents/langgraph/upstream/UPSTREAM_NOTE.md`
for the vendored-source patch list.

Bundled smoke gate:

```bash
bash scripts/smoke_e2e_bundled.sh                  # hermetic
LIVE_E2E=1 bash scripts/smoke_e2e_bundled.sh       # add live vLLM + Qdrant
REME_E2E=1 bash scripts/smoke_e2e_bundled.sh       # add ReMe roundtrip
```
