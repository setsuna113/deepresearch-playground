# Serving setup

Phase 1 routes through two OpenAI-compatible endpoints. Both are served by
vLLM in sidecar virtualenvs outside the playground project.

| Endpoint | Host       | Port | Model                              | GPU plan          |
| -------- | ---------- | ---- | ---------------------------------- | ----------------- |
| local    | WSL        | 8001 | `Qwen/Qwen3.5-9B`                  | 1× RTX 4090 Laptop (16 GB), bitsandbytes 4-bit |
| cloud    | sjtu       | 8000 | `Qwen/Qwen3.6-35B-A3B-FP8`         | 4× RTX 4090, TP=4, FP8 |

The cloud endpoint is reached over an SSH tunnel that forwards
`sjtu:8000` to local `localhost:8002`, so the playground config uses
two `localhost:` URLs.

## One-time setup

```bash
# Local (WSL)
uv venv --python 3.12 /home/lyc/serving/.venv
uv pip install --python /home/lyc/serving/.venv/bin/python vllm bitsandbytes
hf download Qwen/Qwen3.5-9B --local-dir /home/lyc/models/Qwen3.5-9B

# Remote (sjtu) — run on sjtu
uv venv --python 3.12 ~/serving/.venv
uv pip install --python ~/serving/.venv/bin/python vllm
hf download Qwen/Qwen3.6-35B-A3B-FP8 --local-dir ~/models/Qwen3.6-35B-A3B-FP8
```

## Running

In three separate terminals on WSL:

```bash
# 1. Local model server
bash scripts/serve_local.sh

# 2. SSH tunnel to cloud server (which you must launch on sjtu first)
bash scripts/tunnel_sjtu.sh
# (on sjtu, in another shell: bash scripts/serve_cloud.sh)

# 3. The playground
uv run deepresearch run "..." --user alice --project thesis \
  --model-profile co_schedule_v0
```

## Smoke check

```bash
curl -s http://localhost:8001/v1/models | jq '.data[].id'
curl -s http://localhost:8002/v1/models | jq '.data[].id'
```

Both should return the served model id.

## Notes

- The local 16 GB GPU is tight for a 9B model. We rely on bitsandbytes
  4-bit on-the-fly quantization (`--quantization bitsandbytes`). If
  vLLM logs OOM, drop `--max-model-len` or lower `--gpu-memory-utilization`.
- `--enforce-eager` is set locally to avoid CUDA-graph capture, which
  can blow up VRAM during compile. Remove it once a smaller hot path
  proves stable.
- The cloud model uses `--enable-expert-parallel` because Qwen3.6-35B-A3B
  is a 35B/3B MoE — expert parallel sharding matches the TP=4 layout.
