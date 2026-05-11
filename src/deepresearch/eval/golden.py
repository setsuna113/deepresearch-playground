"""Load golden eval suites from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class GoldenItem:
    id: str
    query: str
    notes: str | None = None


def load_suite(path: str | Path) -> list[GoldenItem]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    data = yaml.safe_load(p.read_text()) or {}
    items = []
    for raw in data.get("items", []):
        items.append(GoldenItem(id=str(raw["id"]), query=str(raw["query"]), notes=raw.get("notes")))
    return items
