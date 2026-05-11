"""Prompt templates as plain markdown for easy editing.

Loaded with simple `str.format` (so `{var}` is interpolated, `{{lit}}` is
escaped). The path layout is parallel to the agents module.
"""

from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).parent


def load_prompt(name: str) -> str:
    path = _HERE / f"{name}.md"
    return path.read_text(encoding="utf-8")
