"""Background runner — Phase-1 stub.

Phase 1 schedules orchestrator runs directly with `asyncio.create_task` from
`api/routes/runs.py`; this module is reserved for the queue-backed runner
that Phase 2 will introduce (RQ, Celery, or Temporal) so the API process
isn't responsible for crash recovery.
"""

from __future__ import annotations
