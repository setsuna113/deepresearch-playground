#!/usr/bin/env bash
# Layered end-to-end studio debug runner. Runs L0 → L3 in order, fails fast
# on the first broken layer, keeps the dev server alive at the end so the
# operator can flip to the browser for L4.
#
# Env overrides:
#   QUERY    (default: AWQ vs GPTQ on 4x4090 thesis-flavoured query)
#   PROFILE  (default: co_schedule_v0)
#   SKIP     (comma-separated layer keys to skip: L0,L1,L2,L3)
#
# Prereq: local vLLM running on :8001 (`bash scripts/serve_local.sh` in a
# separate terminal), and `uv sync --extra dev` already run.

set -euo pipefail
cd "$(dirname "$0")/.."

QUERY=${QUERY:-"What are the trade-offs of AWQ vs GPTQ for 70B inference on 4x RTX 4090?"}
PROFILE=${PROFILE:-co_schedule_v0}
SKIP=${SKIP:-}

DEV_PID=""
DEV_LOG=""

cleanup() {
  if [ -n "$DEV_PID" ] && kill -0 "$DEV_PID" 2>/dev/null; then
    echo ""
    echo "stopping dev server (pid=$DEV_PID)"
    kill -TERM "$DEV_PID" 2>/dev/null || true
    wait "$DEV_PID" 2>/dev/null || true
  fi
}

# Always tear down on exit so the dev server doesn't leak on Ctrl-C or
# unexpected failures.
trap cleanup INT TERM EXIT

want() {
  case ",$SKIP," in
    *",$1,"*) return 1 ;;
    *) return 0 ;;
  esac
}

# --- L0 -------------------------------------------------------------------
if want L0; then
  echo ""
  echo "=== L0: preflight ==="
  uv run --extra dev python scripts/preflight_deepseek.py
fi

# --- L1 -------------------------------------------------------------------
if want L1; then
  echo ""
  echo "=== L1: hermetic studio_e2e --fake ==="
  uv run --extra dev python scripts/studio_e2e.py --fake \
      --profile "$PROFILE" --query "$QUERY"
fi

# --- L2 -------------------------------------------------------------------
if want L2; then
  echo ""
  echo "=== L2: live in-process studio_e2e ==="
  uv run --extra dev python scripts/studio_e2e.py \
      --profile "$PROFILE" --query "$QUERY"
fi

# --- L3 -------------------------------------------------------------------
if want L3; then
  echo ""
  echo "=== L3: langgraph dev + sdk driver ==="

  if [ ! -x .venv/bin/langgraph ]; then
    echo "FAIL: .venv/bin/langgraph missing. Run: uv sync --extra dev" >&2
    exit 1
  fi

  mkdir -p logs
  DEV_LOG="logs/studio_dev_$(date +%Y%m%d_%H%M%S).log"
  bash scripts/run_studio_dev.sh > "$DEV_LOG" 2>&1 &
  DEV_PID=$!
  echo "dev pid=$DEV_PID log=$DEV_LOG"

  # Poll /ok up to 90s.  langgraph dev is slow to import sentence-transformers
  # + ReMe on first boot; 90s is conservative.
  ready=0
  for i in $(seq 1 90); do
    if curl -sS -o /dev/null -m 3 http://127.0.0.1:2024/ok; then
      ready=1
      echo "dev server ready (waited ${i}s)"
      break
    fi
    if ! kill -0 "$DEV_PID" 2>/dev/null; then
      echo ""
      echo "FAIL: dev server exited; tail of log:"
      tail -80 "$DEV_LOG" >&2 || true
      exit 1
    fi
    sleep 1
  done
  if [ "$ready" -ne 1 ]; then
    echo ""
    echo "FAIL: dev server did not become ready within 90s; tail of log:"
    tail -80 "$DEV_LOG" >&2 || true
    exit 1
  fi

  if uv run --extra dev python scripts/studio_client_driver.py \
        --url http://127.0.0.1:2024 \
        --query "$QUERY" \
        --profile "$PROFILE"; then
    echo "L3 driver PASS"
  else
    echo ""
    echo "L3 driver FAIL — tail of dev server log:"
    tail -120 "$DEV_LOG" >&2 || true
    exit 1
  fi
fi

# --- L4 prompt ------------------------------------------------------------
if [ -n "$DEV_PID" ]; then
  cat <<EOF

============================================================
 L4: Browser UI (manual)
 Open: https://smith.langchain.com/studio?baseUrl=http%3A%2F%2F127.0.0.1%3A2024
 Pick assistant: 'Deep Researcher (Phase 1.5)'
 Send a query and watch the node tree.
 Press Ctrl-C in THIS terminal to stop the dev server.
============================================================
EOF
  # Disarm EXIT-trap teardown so we wait until SIGINT instead.
  trap - EXIT
  trap cleanup INT TERM
  wait "$DEV_PID"
fi
