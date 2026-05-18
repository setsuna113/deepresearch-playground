from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

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
    max_concurrent: int = typer.Option(2, "--max-concurrent", help="Concurrent researcher subgraphs."),
    model_profile: str = typer.Option("phase1_default", "--model-profile"),
    memory_profile: str = typer.Option("default", "--memory-profile"),
    minimal: bool = typer.Option(
        False,
        "--minimal",
        help="Thesis-baseline preset: caps researcher iterations and concurrency to 1.",
    ),
    out_dir: Path = typer.Option(Path("./data/runs"), "--out-dir"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show step inputs/outputs."),
) -> None:
    """Run a research query end-to-end and write a Markdown report."""

    if minimal:
        max_searches = 1
        max_concurrent = 1

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
            max_concurrent_units=max_concurrent,
            model_profile=model_profile,
            memory_profile=memory_profile,
        )
        run = await run_research(req, deps)
        out_dir.mkdir(parents=True, exist_ok=True)
        run_dir = out_dir / str(run.id)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "report.md").write_text(run.report_md or "(no report)", encoding="utf-8")

        _print_step_table(deps, run.id, verbose=verbose)
        _print_model_call_table(deps, run.id)
        _print_summary(run, run_dir)

        if run.error:
            console.print(f"[red]error[/red]: {run.error}")
            raise typer.Exit(code=1)

    asyncio.run(_go())


def _print_step_table(deps, run_id, *, verbose: bool) -> None:
    steps = sorted(deps.repos.steps.list_for_run(run_id), key=lambda s: s.seq)
    if not steps:
        return
    t = Table(title="Agent steps (what each node did)")
    t.add_column("seq", justify="right")
    t.add_column("role")
    t.add_column("status")
    t.add_column("latency", justify="right")
    t.add_column("llm calls", justify="right")
    t.add_column("parent")
    if verbose:
        t.add_column("output snippet", overflow="fold")
    for s in steps:
        row = [
            str(s.seq),
            s.role.value,
            s.status.value,
            f"{s.latency_ms} ms",
            str(len(s.model_call_ids)),
            str(s.input.get("parent_seq", "")),
        ]
        if verbose:
            row.append(_snippet(s.output))
        t.add_row(*row)
    console.print(t)


def _print_model_call_table(deps, run_id) -> None:
    calls = deps.repos.model_calls.list_for_run(run_id)
    if not calls:
        return
    calls = sorted(calls, key=lambda c: c.started_at)
    t = Table(title="LLM calls (which endpoint Qwen routed to)")
    t.add_column("#", justify="right")
    t.add_column("endpoint")
    t.add_column("model_id")
    t.add_column("role")
    t.add_column("prompt", justify="right")
    t.add_column("completion", justify="right")
    t.add_column("latency", justify="right")
    for i, c in enumerate(calls, 1):
        t.add_row(
            str(i),
            c.endpoint_name,
            c.model_id,
            c.role,
            str(c.prompt_tokens),
            str(c.completion_tokens),
            f"{c.latency_ms} ms",
        )
    console.print(t)


def _print_summary(run, run_dir: Path) -> None:
    console.print(f"\n[bold green]run_id[/bold green] = {run.id}")
    console.print(f"status        = {run.status.value}")
    console.print(f"latency       = {run.metrics.total_latency_ms} ms")
    console.print(f"memory reads  = {run.metrics.n_memory_reads}")
    console.print(f"memory writes = {run.metrics.n_memory_writes}")
    console.print(f"report        = {run_dir / 'report.md'}")


def _snippet(payload: dict, max_chars: int = 120) -> str:
    if not payload:
        return ""
    # Prefer human-readable fields.
    for k in ("final_report", "research_brief", "compressed_research", "messages"):
        if k in payload:
            v = payload[k]
            s = str(v)
            break
    else:
        s = ", ".join(f"{k}=..." for k in payload)[:max_chars]
        return s
    s = s.replace("\n", " ").strip()
    return s[:max_chars] + ("..." if len(s) > max_chars else "")
