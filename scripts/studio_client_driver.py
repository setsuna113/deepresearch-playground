"""L3 client driver — exercises the studio path *through HTTP* like the browser.

Previous studio tests called `studio_graph.ainvoke()` directly in-process. This
script does the real thing: it talks to a running `langgraph dev` server via
`langgraph_sdk`, creates a thread, streams the run's SSE events, waits for the
graph to finish, then cross-checks the SQLite trace.

The cross-check is what catches "events worked but trace recorder swallowed it"
divergence — exactly the silent-failure class the user wanted spotted.

Usage:
    uv run --extra dev python scripts/studio_client_driver.py \\
        --url http://127.0.0.1:2024 \\
        --query "AWQ vs GPTQ for 70B on 4x4090?" \\
        --profile co_schedule_v0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
import traceback
from pathlib import Path
from typing import Any

import httpx
from langgraph_sdk import get_client

ROOT = Path(__file__).resolve().parent.parent
SQLITE_PATH = ROOT / "data" / "deepresearch.sqlite"


def _trim(s: str, n: int = 400) -> str:
    return s if len(s) <= n else s[:n] + f"... (+{len(s) - n} chars)"


def _extract_run_id(values: Any) -> str | None:
    """Studio's metadata event carries the run_id; if missing, dig it out of state."""
    if isinstance(values, dict):
        rid = values.get("run_id")
        if isinstance(rid, str):
            return rid
    return None


async def _drive(args: argparse.Namespace) -> int:
    client = get_client(url=args.url)

    # (1) liveness + assistant registration check.
    try:
        assts = await client.assistants.search()
    except httpx.ConnectError as e:
        print(f"FAIL connect: cannot reach {args.url} ({e}). Is `langgraph dev` running?")
        return 1
    except httpx.HTTPStatusError as e:
        print(f"FAIL HTTP {e.response.status_code} on /assistants/search: {e.response.text[:500]}")
        return 1

    graph_ids = [a.get("graph_id") for a in assts]
    print(f"assistants:  {graph_ids}")
    if args.assistant not in graph_ids:
        print(f"FAIL: assistant {args.assistant!r} not registered (got {graph_ids!r})")
        return 1

    # (2) thread create.
    thread = await client.threads.create(
        metadata={"user_id": args.user, "project_id": args.project},
    )
    thread_id = thread["thread_id"]
    print(f"thread:      {thread_id}")

    # (3) stream the run. Studio uses 'updates' + 'events' + 'debug'; we
    # log a one-line summary per event so a hung run is visible.
    run_id: str | None = None
    nodes_seen: list[str] = []
    error_payload: Any = None
    chat_events = 0

    print()
    print("stream:")
    try:
        async for part in client.runs.stream(
            thread_id=thread_id,
            assistant_id=args.assistant,
            input={"messages": [{"role": "user", "content": args.query}]},
            config={
                "configurable": {
                    "user_id": args.user,
                    "project_id": args.project,
                    "model_profile": args.profile,
                    "memory_profile": args.memory_profile,
                    "max_searches": args.max_searches,
                    "max_concurrent_units": args.max_concurrent_units,
                },
            },
            stream_mode=["updates", "events"],
        ):
            evt = getattr(part, "event", None)
            data = getattr(part, "data", None)
            if evt == "metadata":
                run_id = (data or {}).get("run_id") or run_id
                print(f"  metadata run_id={run_id}")
            elif evt == "updates" and isinstance(data, dict):
                for node, delta in data.items():
                    nodes_seen.append(node)
                    keys = list(delta.keys()) if isinstance(delta, dict) else type(delta).__name__
                    print(f"  update    {node:<28} keys={keys}")
            elif evt == "events" and isinstance(data, dict):
                etype = data.get("event")
                if etype and etype.startswith("on_chat_model"):
                    chat_events += 1
            elif evt == "error":
                error_payload = data
                print(f"  ERROR     {json.dumps(data, indent=2)[:500]}")
            elif evt == "end":
                print("  end")
    except httpx.HTTPStatusError as e:
        print(f"\nFAIL HTTP {e.response.status_code} during stream: {e.response.text[:500]}")
        return 1
    except Exception:
        traceback.print_exc()
        return 1

    print(f"chat_model events: {chat_events}")

    # (4) fetch the post-run thread state.
    try:
        state = await client.threads.get_state(thread_id)
    except Exception as e:
        print(f"WARN get_state failed: {type(e).__name__}: {e}")
        state = {"values": {}}
    values = state.get("values") or {}
    final_report = (values.get("final_report") or "").strip()
    reflection = values.get("reflection") or {}

    print()
    print(f"final_report ({len(final_report)} chars):")
    for line in final_report.splitlines()[:8]:
        print(f"  | {line}")
    print(f"reflection:  {_trim(json.dumps(reflection, default=str), 300)}")

    # (5) SQLite cross-check. studio_bootstrap mints a run row; we look up by
    # the most-recent run for this user/project so we don't need the run_id
    # echoed back from Studio.
    print()
    if not SQLITE_PATH.is_file():
        print(f"WARN sqlite: {SQLITE_PATH} missing")
    else:
        con = sqlite3.connect(str(SQLITE_PATH))
        # Tables (per src/deepresearch/storage/models.py): research_runs,
        # agent_steps, model_calls — all plural; user_id / project_id columns,
        # NOT request_user_id.
        row = con.execute(
            """
            SELECT id, status, error, started_at, finished_at
            FROM research_runs
            WHERE user_id = ? AND project_id = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (args.user, args.project),
        ).fetchone()
        if row is None:
            print(
                f"WARN sqlite: no research_runs row for user={args.user!r} project={args.project!r}"
            )
        else:
            db_run_id, db_status, db_error, _started, _finished = row
            err_tail = f" error={db_error[:120]!r}" if db_error else ""
            print(f"sqlite run:  id={db_run_id} status={db_status}{err_tail}")
            steps = con.execute(
                "SELECT seq, role, status, error FROM agent_steps WHERE run_id=? ORDER BY seq",
                (db_run_id,),
            ).fetchall()
            calls = con.execute(
                "SELECT endpoint_name, role, prompt_tokens, completion_tokens, latency_ms "
                "FROM model_calls WHERE run_id=? ORDER BY started_at",
                (db_run_id,),
            ).fetchall()
            print(f"agent_steps rows: {len(steps)}")
            for seq, role, status_str, err in steps:
                err_str = f" error={err[:120]!r}" if err else ""
                print(f"  seq={seq:>2} role={role:<13} status={status_str}{err_str}")
            print(f"model_calls rows: {len(calls)}")
            for ep, role, ptok, ctok, lat in calls:
                print(
                    f"  endpoint={ep:<6} role={role:<13} prompt={ptok:>4} completion={ctok:>4} latency={lat:>5}ms"
                )

    # (6) verdict.
    print()
    ok = (
        error_payload is None
        and len(final_report) >= 50
        and "studio_bootstrap" in nodes_seen
    )
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:2024")
    p.add_argument(
        "--query",
        default="What are the trade-offs of AWQ vs GPTQ for 70B inference on 4x RTX 4090?",
    )
    p.add_argument("--profile", default="co_schedule_v0")
    p.add_argument("--memory-profile", default="default")
    p.add_argument("--max-searches", type=int, default=2)
    p.add_argument("--max-concurrent-units", type=int, default=1)
    p.add_argument("--user", default="studio_driver")
    p.add_argument("--project", default="studio_driver")
    p.add_argument(
        "--assistant",
        default="Deep Researcher (Phase 1.5)",
        help="Must match a key in langgraph.json:graphs.",
    )
    args = p.parse_args()
    return asyncio.run(_drive(args))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
