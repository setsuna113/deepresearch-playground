from __future__ import annotations

import asyncio

import typer
from rich.console import Console

from deepresearch.api.deps import build_dependencies
from deepresearch.config import get_config
from deepresearch.schemas.memory import MemoryType

console = Console()


def cmd_memory_query(
    query: str = typer.Argument(..., help="Query string for memory retrieval."),
    user: str = typer.Option("default", "--user", "-u"),
    project: str = typer.Option("default", "--project", "-p"),
    type: MemoryType = typer.Option(MemoryType.procedural, "--type"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
) -> None:
    """Query memory directly."""

    async def _go() -> None:
        cfg = get_config()
        deps = await build_dependencies(cfg)
        records = await deps.memory.query(
            run_id=None,
            user_id=user,
            project_id=project,
            query=query,
            memory_type=type,
            top_k=top_k,
        )
        if not records:
            console.print("[yellow]no matches[/yellow]")
            return
        for i, r in enumerate(records, start=1):
            score = f" [{r.score:.3f}]" if r.score is not None else ""
            console.print(f"[bold]{i}.[/bold]{score} {r.content}")

    asyncio.run(_go())
