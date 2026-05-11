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

## Phase 1 (current)

A STORM-style DeepResearch loop with **ReMe** as the primary memory layer:

- Agents: planner, searcher, reader, synthesizer, reflector.
- Memory: personal / task / tool via ReMe (Qdrant backend), plus a
  working-memory collection that we manage ourselves.
- Storage: SQLite for runs/traces/metrics.
- Runtime: FastAPI service + Typer CLI.
- All LLM calls go through an OpenAI-compatible `ModelClient`.

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

## Layout

```
src/deepresearch/
  schemas/        # pydantic v2 data shapes
  config/         # YAML + env config loader
  storage/        # SQLite (SQLModel) tables and repos
  models/         # OpenAI-compatible client + Router (Phase-4 seam)
  memory/         # ReMe adapter + working memory
  tools/          # search providers + page fetch
  agents/         # planner/searcher/reader/synthesizer/reflector + orchestrator
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

Phase 1 skeleton. See `/home/lyc/.claude/plans/local-project-dir-new-synthetic-robin.md`
for the full phased plan.
