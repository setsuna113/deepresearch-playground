"""Run a golden eval suite end-to-end, with and without memory."""

from __future__ import annotations

import time
from pathlib import Path

from deepresearch.agents.context import RunDependencies
from deepresearch.agents.orchestrator import run_research
from deepresearch.eval.golden import load_suite
from deepresearch.schemas.runs import RunRequest


async def run_suite(
    *,
    suite: str,
    deps: RunDependencies,
    out_path: Path,
    user: str,
    project: str,
) -> None:
    suite_path = deps.config.eval.golden_suite_path if suite == "golden" else suite
    items = load_suite(suite_path)
    rows: list[dict] = []
    for item in items:
        for memory_profile in ("default", "none"):
            t0 = time.perf_counter()
            req = RunRequest(
                query=item.query,
                user_id=f"{user}-{memory_profile}",
                project_id=project,
                memory_profile=memory_profile,
                max_searches=3,
            )
            run = await run_research(req, deps)
            rows.append(
                {
                    "id": item.id,
                    "memory_profile": memory_profile,
                    "status": run.status.value,
                    "report_chars": len(run.report_md or ""),
                    "n_citations": len(run.citations),
                    "latency_ms": run.metrics.total_latency_ms,
                    "wall_ms": int((time.perf_counter() - t0) * 1000),
                    "error": run.error,
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Eval: phase1\n", "| id | memory | status | chars | cites | latency_ms | error |",
             "|----|--------|--------|-------|-------|-----------|-------|"]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['memory_profile']} | {r['status']} | {r['report_chars']} | "
            f"{r['n_citations']} | {r['latency_ms']} | {r['error'] or ''} |"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


async def _noop() -> None:
    """Imported by eval/__init__ to keep the module compile-clean."""
    return None
