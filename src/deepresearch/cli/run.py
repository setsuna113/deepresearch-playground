from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from deepresearch.agents.orchestrator import run_research
from deepresearch.api.deps import build_dependencies
from deepresearch.config import get_config
from deepresearch.observability.logging import configure_logging
from deepresearch.schemas.runs import Depth, RunRequest

console = Console()


def cmd_run(
    query: str = typer.Argument(..., help="Research question."),
    user: str = typer.Option("default", "--user", "-u"),
    project: str = typer.Option("default", "--project", "-p"),
    depth: Depth = typer.Option(Depth.standard, "--depth"),
    max_searches: int = typer.Option(5, "--max-searches"),
    model_profile: str = typer.Option("phase1_default", "--model-profile"),
    memory_profile: str = typer.Option("default", "--memory-profile"),
    out_dir: Path = typer.Option(Path("./data/runs"), "--out-dir"),
) -> None:
    """Run a research query end-to-end and write a Markdown report."""

    async def _go() -> None:
        cfg = get_config()
        configure_logging(level=cfg.app.log_level, json=cfg.app.log_json)
        deps = await build_dependencies(cfg)
        req = RunRequest(
            query=query,
            user_id=user,
            project_id=project,
            depth=depth,
            max_searches=max_searches,
            model_profile=model_profile,
            memory_profile=memory_profile,
        )
        run = await run_research(req, deps)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_dir = out_dir / str(run.id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "report.md").write_text(run.report_md or "(no report)", encoding="utf-8")
        console.print(f"[bold green]run_id[/bold green] = {run.id}")
        console.print(f"status = {run.status.value}")
        console.print(f"report = {run_dir / 'report.md'}")
        if run.error:
            console.print(f"[red]error[/red]: {run.error}")
            raise typer.Exit(code=1)

    asyncio.run(_go())
