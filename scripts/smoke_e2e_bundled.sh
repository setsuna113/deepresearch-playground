#!/usr/bin/env bash
# Phase 1.5 bundled smoke gate.
#
# Goes beyond the original `smoke_e2e.sh` by adding:
#   - Gate 6: ReMe roundtrip (skipped unless REME_E2E=1 — requires a
#             live embedding endpoint and Qdrant).
#   - Gate 7: parallel researcher dispatch — verified via the
#             unit/integration tests rather than a live model, since
#             our fake-client demo already exercises that path.
#
# The fake-client demo runs unconditionally — it is hermetic (no
# vLLM, no Tavily, no ReMe) and exercises the full pipeline:
# clarify -> brief -> supervisor -> final_report -> reflector with
# both local and cloud endpoints in the routing trace.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> 1/6 Hermetic e2e demo (fake LLM, fake search, embedded Qdrant)"
uv run --extra dev python scripts/demo_e2e.py --profile co_schedule_v0 >/tmp/demo_e2e.out 2>&1
if ! grep -q "^  RESULT: PASS" /tmp/demo_e2e.out; then
  echo "FAIL: demo_e2e.py did not report PASS"
  cat /tmp/demo_e2e.out
  exit 1
fi
echo "    demo: PASS (see /tmp/demo_e2e.out)"

echo "==> 2/6 Unit + integration tests (36 tests)"
uv run --extra dev pytest --tb=short -q

echo "==> 3/6 Linter"
uv run --extra dev ruff check src tests scripts/demo_e2e.py

# The remaining gates require live infrastructure. They run only when
# the operator opts in via env flags.
if [[ "${LIVE_E2E:-0}" == "1" ]]; then
  echo "==> 4/6 Live Qdrant + vLLM smoke (LIVE_E2E=1)"
  bash scripts/run_qdrant_local.sh
  uv run deepresearch db init
  uv run deepresearch run \
    "Trade-offs of AWQ vs GPTQ for 70B inference on 4x RTX 4090?" \
    --user alice --project thesis --depth quick --max-searches 2
  uv run deepresearch run "AWQ vs GPTQ trade-offs again" \
    --user alice --project thesis --depth quick --max-searches 2
  uv run deepresearch memory query "AWQ" --user alice --project thesis --type task
  uv run deepresearch run "AWQ vs GPTQ trade-offs" \
    --user alice --project thesis --depth quick --memory-profile none --max-searches 2
  uv run deepresearch eval --suite golden --report ./data/eval/phase1.md
else
  echo "==> 4/6 SKIPPED — live vLLM+Qdrant gate (set LIVE_E2E=1 to run)"
fi

if [[ "${LIVE_REME:-${REME_E2E:-0}}" == "1" ]]; then
  echo "==> 5/6 ReMe roundtrip (LIVE_REME=1)"
  # Mocked unit tests first (catches structural regressions).
  uv run --extra dev pytest tests/test_reme_adapter.py -v
  # Then the live roundtrip — talks to a real LLM + embedding endpoint.
  # Skips internally if creds/endpoint envs aren't set.
  LIVE_REME=1 uv run --extra dev pytest tests/test_reme_live.py -v
else
  echo "==> 5/6 SKIPPED — ReMe roundtrip (set LIVE_REME=1 to run)"
fi

echo "==> 6/6 Bundled smoke passed."
