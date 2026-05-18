#!/usr/bin/env bash
# Launch the LangGraph Studio dev server in the foreground.
#
# Replaces the docstring-only invocation in src/deepresearch/agents/langgraph/studio.py
# so the same launch can be used by debug_studio.sh AND by humans running the
# server interactively for the browser UI.
#
# Env overrides:
#   STUDIO_PORT (default 2024)  STUDIO_HOST (default 127.0.0.1)
#
# Logs both to stdout (so the parent sees crashes immediately) AND to a
# timestamped file under logs/.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

PORT=${STUDIO_PORT:-2024}
HOST=${STUDIO_HOST:-127.0.0.1}
LOG="logs/studio_dev_$(date +%Y%m%d_%H%M%S).log"

# Cheap input validation — every failure should name the missing file.
for f in langgraph.json .env config/config.local.yaml; do
  if [ ! -f "$f" ]; then
    echo "MISSING: $f (run from repo root)" >&2
    exit 1
  fi
done

echo "Studio dev server: http://${HOST}:${PORT}"
echo "Browser URL:        https://smith.langchain.com/studio?baseUrl=http%3A%2F%2F${HOST}%3A${PORT}"
echo "Log:                ${LOG}"
echo "Press Ctrl-C to stop."
echo ""

# Prefer the .venv binary (installed via `uv sync --extra dev`); fall back
# to uvx for transient installs.
if [ -x .venv/bin/langgraph ]; then
  # --no-reload + --no-browser: ReMe writes to ./data/reme/ constantly,
  # which makes the default watchfiles loop emit "changes detected" every
  # second and starves the event loop. --no-reload disables the watchdog
  # entirely. The composite runner already manages process lifecycle.
  exec .venv/bin/langgraph dev \
       --host "$HOST" --port "$PORT" --allow-blocking --no-reload --no-browser \
       2>&1 | tee -a "$LOG"
elif command -v uvx >/dev/null 2>&1; then
  exec uvx --from "langgraph-cli[inmem]" --with-editable . --python 3.12 \
       langgraph dev --host "$HOST" --port "$PORT" --allow-blocking --no-reload --no-browser \
       2>&1 | tee -a "$LOG"
else
  echo "langgraph-cli not installed; run: uv sync --extra dev" >&2
  exit 1
fi
