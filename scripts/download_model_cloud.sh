#!/usr/bin/env bash
# Download a model to the sjtu serving node.
#
# huggingface.co is blocked on sjtu (GFW timeout). Two mirrors are
# reachable; we prefer ModelScope (Alibaba CDN, faster for Qwen models)
# and fall back to hf-mirror.com if `modelscope` isn't installed or the
# model isn't mirrored there.
#
# Usage:
#   bash scripts/download_model_cloud.sh [MODEL] [DEST]
#
# Defaults:
#   MODEL=Qwen/Qwen3.6-35B-A3B-FP8
#   DEST=$HOME/models/$(basename MODEL)
#
# On sjtu, `~/.bashrc` already exports HF_ENDPOINT=https://hf-mirror.com,
# so the HF fallback works out of the box. ModelScope needs
#   pip install modelscope
# inside the serving venv (one-time setup).

set -euo pipefail

MODEL="${1:-Qwen/Qwen3.6-35B-A3B-FP8}"
DEST="${2:-$HOME/models/$(basename "$MODEL")}"
SERVING_VENV="${SERVING_VENV:-$HOME/serving/.venv}"

mkdir -p "$(dirname "$DEST")"

if [ -d "${SERVING_VENV}" ]; then
  # Prefer the serving venv's tools so we use the same env that vLLM
  # later imports the model with.
  PATH="${SERVING_VENV}/bin:${PATH}"
fi

if command -v modelscope >/dev/null 2>&1; then
  echo "==> Downloading ${MODEL} via ModelScope → ${DEST}"
  modelscope download --model "${MODEL}" --local_dir "${DEST}"
  exit 0
fi

if command -v hf >/dev/null 2>&1; then
  echo "==> ModelScope not installed; falling back to hf-mirror.com"
  echo "    (install ModelScope for typically faster downloads: pip install modelscope)"
  HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
    hf download "${MODEL}" --local-dir "${DEST}"
  exit 0
fi

echo "ERROR: neither 'modelscope' nor 'hf' (huggingface-cli) is available." >&2
echo "Install one in ${SERVING_VENV}:" >&2
echo "  ${SERVING_VENV}/bin/pip install modelscope" >&2
echo "    or" >&2
echo "  ${SERVING_VENV}/bin/pip install 'huggingface_hub[cli]'" >&2
exit 1
