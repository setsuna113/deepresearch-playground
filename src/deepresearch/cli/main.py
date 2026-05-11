"""Typer CLI entry. `uv run deepresearch ...` or `python -m deepresearch.cli.main`."""

from __future__ import annotations

import typer

from deepresearch.cli import db as db_cmd
from deepresearch.cli import eval as eval_cmd
from deepresearch.cli import memory as memory_cmd
from deepresearch.cli import run as run_cmd

app = typer.Typer(no_args_is_help=True, add_completion=False, help="deepresearch CLI")
app.command(name="run", help="Run a research query end-to-end.")(run_cmd.cmd_run)
app.command(name="eval", help="Run a named eval suite.")(eval_cmd.cmd_eval)

memory_app = typer.Typer(no_args_is_help=True, help="Memory operations.")
memory_app.command(name="query")(memory_cmd.cmd_memory_query)
app.add_typer(memory_app, name="memory")

db_app = typer.Typer(no_args_is_help=True, help="Database operations.")
db_app.command(name="init")(db_cmd.cmd_db_init)
app.add_typer(db_app, name="db")


if __name__ == "__main__":
    app()
