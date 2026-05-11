#!/usr/bin/env bash
# Serve Qwen3.5-9B via vLLM on the local WSL 4090 Laptop (16 GB).
# 9B BF16 won't fit; we quantize to 4-bit at load time via bitsandbytes.
set -euo pipefail

SERVING_VENV=${SERVING_VENV:-/home/lyc/serving/.venv}
MODEL=${MODEL:-/home/lyc/models/Qwen3.5-9B}
PORT=${PORT:-8001}
MAX_LEN=${MAX_LEN:-4096}
GPU_UTIL=${GPU_UTIL:-0.85}

if [ ! -x "${SERVING_VENV}/bin/python" ]; then
  echo "serving venv missing at ${SERVING_VENV}; run scripts/setup_serving_local.sh first" >&2
  exit 1
fi
if [ ! -d "${MODEL}" ]; then
  echo "model missing at ${MODEL}; run: hf download Qwen/Qwen3.5-9B --local-dir ${MODEL}" >&2
  exit 1
fi

exec "${SERVING_VENV}/bin/vllm" serve "${MODEL}" \
  --port "${PORT}" \
  --host 127.0.0.1 \
  --served-model-name Qwen/Qwen3.5-9B \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --enforce-eager \
  --dtype bfloat16
