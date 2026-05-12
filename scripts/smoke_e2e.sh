#!/usr/bin/env bash
# Phase-1 smoke gate. Exits non-zero on first failure.
set -euo pipefail

cd "$(dirname "$0")/.."

# 0. infra
bash scripts/run_qdrant_local.sh
uv run deepresearch db init

# 1. CLI happy path (depends on a local OpenAI-compatible endpoint at localhost:8001
#    and a TAVILY_API_KEY in .env — both expected to be configured for the smoke gate).
uv run deepresearch run \
  "Trade-offs of AWQ vs GPTQ for 70B inference on 4x RTX 4090?" \
  --user alice --project thesis --depth quick --max-searches 2

# 2. Memory roundtrip — repeat-query reuse check
uv run deepresearch run "AWQ vs GPTQ trade-offs again" \
  --user alice --project thesis --depth quick --max-searches 2

uv run deepresearch memory query "AWQ" --user alice --project thesis --type task

# 3. No-memory ablation
uv run deepresearch run "AWQ vs GPTQ trade-offs" \
  --user alice --project thesis --depth quick --memory-profile none --max-searches 2

# 4. Golden suite
uv run deepresearch eval --suite golden --report ./data/eval/phase1.md

echo "smoke gate passed."
