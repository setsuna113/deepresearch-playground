#!/usr/bin/env bash
# Serve Qwen3-8B-AWQ via vLLM on the local WSL 4090 Laptop (16 GB).
# The 16 GB GPU is shared with Windows-side apps (~8 GB held outside WSL's
# view), so we have ~8 GB usable. AWQ-quantized Qwen3-8B (~5 GB weights)
# plus a small KV cache fits.
#
# Note: We originally tried Qwen/Qwen3.5-9B but it's a vision-language
# model whose linear-attention text trunk plus vision encoder doesn't
# fit in the available VRAM and overflows WSL's 15 GB system RAM during
# bitsandbytes quantization. Qwen3-8B-AWQ is the closest text-only,
# pre-quantized fit.
set -euo pipefail

SERVING_VENV=${SERVING_VENV:-/home/lyc/serving/.venv}
MODEL=${MODEL:-/home/lyc/models/Qwen3-8B-AWQ}
PORT=${PORT:-8001}
MAX_LEN=${MAX_LEN:-4096}
GPU_UTIL=${GPU_UTIL:-0.85}

if [ ! -x "${SERVING_VENV}/bin/python" ]; then
  echo "serving venv missing at ${SERVING_VENV}" >&2
  exit 1
fi
if [ ! -d "${MODEL}" ]; then
  echo "model missing at ${MODEL}; run: hf download Qwen/Qwen3-8B-AWQ --local-dir ${MODEL}" >&2
  exit 1
fi

exec "${SERVING_VENV}/bin/vllm" serve "${MODEL}" \
  --port "${PORT}" \
  --host 127.0.0.1 \
  --served-model-name Qwen/Qwen3-8B-AWQ \
  --quantization awq \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --enforce-eager \
  --dtype float16 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
