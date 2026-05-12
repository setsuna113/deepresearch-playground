from __future__ import annotations

from rich.console import Console

from deepresearch.config import get_config
from deepresearch.storage.db import init_db

console = Console()


def cmd_db_init() -> None:
    """Create SQLite tables if missing."""
    cfg = get_config()
    init_db(cfg.app.sqlite_path)
    console.print(f"[green]initialized[/green] sqlite at {cfg.app.sqlite_path}")
