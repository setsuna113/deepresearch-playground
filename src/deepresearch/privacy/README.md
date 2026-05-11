# privacy/ — Phase 4-5 placeholder

This package will hold the contextual-integrity (CI) filter and the
leakage-proxy metrics that ParetoDispatch and the Reflection Broadcast
Protocol depend on.

## Seam

`PrivacyEnvelope` (already at `schemas/privacy.py`) is attached to every
user message, memory, search result, reflection, and model call. In Phase 1
all envelopes default to `{level: public, source: unknown}`.

Phase 4 introduces:

1. A **classifier** that fills in `subject / sender / recipient / attribute /
   transmission_principle / sensitivity_score` from text.
2. **Leakage proxies**: sensitive-token exposure, entity exposure, count of
   cloud-visible memories, mutual-information budget proxy.
3. A **sanitizer** that produces a minimized payload before any cloud call.

## Files (planned)

- `classifier.py`
- `sanitizer.py`
- `leakage.py` — the proxy metrics
- `budgets.py` — per-user MI budget tracking

## Phase-1 invariants to preserve

- Don't break the `PrivacyEnvelope` schema.
- Memory writes already pass through `MemoryService.write()` — that's the
  natural choke point for CI checks later.
