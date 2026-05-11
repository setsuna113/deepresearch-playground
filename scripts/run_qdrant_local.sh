#!/usr/bin/env bash
# Start a local Qdrant for ReMe + working memory.
set -euo pipefail

NAME=${QDRANT_CONTAINER:-dr-qdrant}
PORT=${QDRANT_PORT:-6333}
DATA_DIR=${QDRANT_DATA_DIR:-$(pwd)/data/qdrant}

mkdir -p "$DATA_DIR"

if docker ps --format '{{.Names}}' | grep -q "^${NAME}\$"; then
  echo "qdrant already running as ${NAME}"
  exit 0
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${NAME}\$"; then
  docker start "$NAME"
else
  docker run -d \
    --name "$NAME" \
    -p "${PORT}":6333 \
    -p "$((PORT + 1))":6334 \
    -v "$DATA_DIR":/qdrant/storage \
    qdrant/qdrant:latest
fi

echo "qdrant running at http://localhost:${PORT}"
