# scheduling/ — Phase 4 placeholder

This package will hold **ParetoDispatch**: privacy-bounded co-scheduling
across the local 4090 and the 4×4090 cloud node.

## Seam

`models/router.py` exposes `Router.select(profile, role, envelope, hint)`.
In Phase 1 the body ignores `envelope` and `hint`. Phase 4 swaps in a
ParetoDispatcher that:

1. Reads the current `PrivacyEnvelope` and any `SchedulingHint` (deadline,
   expected cost, mutual-information budget remaining).
2. Reads live GPU utilization from local NVML and a cloud-side metrics
   endpoint.
3. Picks the Pareto-optimal endpoint across `(latency, leakage, util)`.

## Files (planned)

- `dispatcher.py` — `class ParetoDispatcher(Router)` with the same `select()` signature.
- `metrics.py` — local NVML + remote DCGM scrape.
- `policies.py` — the optimization itself (weighted sum / lex / Bayes).

## Phase-1 invariants to preserve

- Don't change `Router.select()`'s signature.
- Every `ModelCallRecord` already carries a `PrivacyEnvelope` — use that.
- The orchestrator must not import from `scheduling/` directly; it should
  receive a `Router` (or a `Router` subclass) via dependency injection.
