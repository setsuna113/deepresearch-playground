from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from deepresearch.api.deps import build_dependencies
from deepresearch.config import get_config
from deepresearch.eval.runner import run_suite

console = Console()


def cmd_eval(
    suite: str = typer.Option("golden", "--suite"),
    report: Path = typer.Option(Path("./data/eval/phase1.md"), "--report"),
    user: str = typer.Option("evaluator", "--user"),
    project: str = typer.Option("eval", "--project"),
) -> None:
    """Run an eval suite and write a Markdown summary."""

    async def _go() -> None:
        cfg = get_config()
        deps = await build_dependencies(cfg)
        await run_suite(suite=suite, deps=deps, out_path=report, user=user, project=project)
        console.print(f"[green]eval written[/green] {report}")

    asyncio.run(_go())
