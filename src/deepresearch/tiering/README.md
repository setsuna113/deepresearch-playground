# tiering/ — Phase 5 placeholder

This package will hold the **hot / warm / cold hierarchical memory
placement** policy across local-GPU, local-SSD, and cloud tiers.

## Seam

`memory/service.py::MemoryService` is the only caller of `ReMeAdapter` and
`WorkingMemory`. Phase 5 wraps it: `TieredMemoryService` consults a
placement policy on every read/write and picks the right backend.

## Files (planned)

- `placement.py` — policy (recency, utility, sensitivity, retrieval freq,
  expected token savings).
- `mover.py` — promotion/demotion between tiers.
- `tiers.py` — `HotTier` (local in-context summaries), `WarmTier` (local
  ReMe/Qdrant), `ColdTier` (cloud-side stores).

## Phase-1 invariants to preserve

- `MemoryService` must keep its query/write surface.
- All `MemoryEvent` rows must carry a `backend` field — that's the
  read-path signal Phase 5 uses to verify tier decisions.
