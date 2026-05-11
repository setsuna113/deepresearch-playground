#!/usr/bin/env bash
# Serve Qwen3.6-35B-A3B-FP8 on sjtu (4x RTX 4090) via vLLM with TP=4.
# FP8 fits well — ~35 GB total weights distributed across 4 GPUs.
set -euo pipefail

SERVING_VENV=${SERVING_VENV:-$HOME/serving/.venv}
MODEL=${MODEL:-$HOME/models/Qwen3.6-35B-A3B-FP8}
PORT=${PORT:-8000}
MAX_LEN=${MAX_LEN:-8192}
TP=${TP:-4}

if [ ! -x "${SERVING_VENV}/bin/python" ]; then
  echo "serving venv missing at ${SERVING_VENV}" >&2
  exit 1
fi
if [ ! -d "${MODEL}" ]; then
  echo "model missing at ${MODEL}" >&2
  exit 1
fi

exec "${SERVING_VENV}/bin/vllm" serve "${MODEL}" \
  --port "${PORT}" \
  --host 0.0.0.0 \
  --served-model-name Qwen/Qwen3.6-35B-A3B-FP8 \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization 0.92 \
  --enable-expert-parallel
