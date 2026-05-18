#!/usr/bin/env bash
# Serve Qwen3.6-35B-A3B-FP8 on sjtu (4x RTX 4090) via vLLM with TP=4.
# FP8 fits well — ~35 GB total weights distributed across 4 GPUs.
#
# Quirks worth remembering:
#
# 1. **torch.compile OOM on 4090s.** The previous run on this model
#    OOM'd during torch.compile/warmup, leaving dangling CUDA
#    allocations that survived container restart (host-level GPU reset
#    was needed). `--enforce-eager` skips the compile pass entirely;
#    the latency hit is small for a MoE model that already activates
#    only 3 B params per token. We trade a slightly slower steady-state
#    for a *much* more recoverable warmup.
#
# 2. **KV cache headroom.** Default --gpu-memory-utilization 0.92 left
#    no slack for KV cache spikes once compile artifacts were resident.
#    We drop to 0.85 to give vLLM room to grow without OOMing.
#
# 3. **Max model len.** Upstream default preallocates KV for 128 k
#    context. Our agentic workflow never approaches that, so we cap at
#    32 k — frees substantial KV cache budget.
set -euo pipefail

SERVING_VENV=${SERVING_VENV:-$HOME/serving/.venv}
MODEL=${MODEL:-$HOME/models/Qwen3.6-35B-A3B-FP8}
PORT=${PORT:-8000}
MAX_LEN=${MAX_LEN:-32768}
TP=${TP:-4}
GPU_UTIL=${GPU_UTIL:-0.85}

if [ ! -x "${SERVING_VENV}/bin/python" ]; then
  echo "serving venv missing at ${SERVING_VENV}" >&2
  exit 1
fi
if [ ! -d "${MODEL}" ]; then
  echo "model missing at ${MODEL}" >&2
  echo "Download with: bash scripts/download_model_cloud.sh" >&2
  exit 1
fi

exec "${SERVING_VENV}/bin/vllm" serve "${MODEL}" \
  --port "${PORT}" \
  --host 0.0.0.0 \
  --served-model-name Qwen/Qwen3.6-35B-A3B-FP8 \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --enforce-eager \
  --enable-expert-parallel \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
