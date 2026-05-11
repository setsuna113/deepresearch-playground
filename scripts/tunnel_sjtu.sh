#!/usr/bin/env bash
# Forward sjtu:8000 -> localhost:8002 so the playground can reach the
# cloud vLLM through a single localhost-only URL.
set -euo pipefail

REMOTE_PORT=${REMOTE_PORT:-8000}
LOCAL_PORT=${LOCAL_PORT:-8002}
HOST=${HOST:-sjtu}

exec ssh -N -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" "${HOST}"
