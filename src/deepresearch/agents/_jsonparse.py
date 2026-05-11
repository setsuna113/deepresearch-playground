"""Best-effort JSON extraction from LLM completions.

Models often wrap JSON in ```json fences, prepend a sentence, or include a
trailing comma. We try plain `json.loads`, then a fenced-block extract,
then a brace-balanced extraction.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL)


def _find_balanced(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_json(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    block = _find_balanced(text)
    if block:
        return json.loads(block)
    raise ValueError(f"could not parse JSON from: {text[:200]!r}")
