# broadcast/ — Phase 5 placeholder

This package will hold the **Reflection Broadcast Protocol**: CI-filtered
propagation of reflections (e.g. "user dislikes Wikipedia") from the local
reflector back to local memory AND laterally to cloud subagents, under a
mutual-information budget.

## Seam

`schemas/agents.py::ReflectionUpdate` already carries an optional
`broadcast_candidate: BroadcastCandidate | None`. The Reflector emits it;
in Phase 1 nothing consumes it.

## Files (planned)

- `protocol.py` — the broadcast logic; consumes `BroadcastCandidate` events.
- `budget.py` — tracks the per-session MI budget.
- `transport.py` — pushes accepted broadcasts to cloud subagents.

## Phase-1 invariants to preserve

- Don't break the `BroadcastCandidate` schema.
- The Reflector must continue to emit it; Phase-1 reflector code path
  populates `broadcast_candidate=None` and that's fine.
