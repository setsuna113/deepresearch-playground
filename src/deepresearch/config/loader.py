"""Load config.yaml with env-var interpolation.

Supports ${VAR} and ${VAR:-default} syntax. Reads config.local.yaml if
present (gitignored), otherwise config.example.yaml.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from deepresearch.config.schema import AppConfig

_ENV_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def _interpolate(value: Any) -> Any:
    if isinstance(value, str):
        def repl(m: re.Match[str]) -> str:
            var, default = m.group(1), m.group(2)
            return os.environ.get(var, default if default is not None else "")

        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def _resolve_path(explicit: str | os.PathLike[str] | None) -> Path:
    if explicit:
        return Path(explicit)
    env_path = os.environ.get("DR_CONFIG")
    if env_path:
        return Path(env_path)
    # Look up from CWD; prefer local override, fall back to example.
    here = Path.cwd()
    for candidate in (
        here / "config" / "config.local.yaml",
        here / "config" / "config.example.yaml",
    ):
        if candidate.exists():
            return candidate
    # Final fallback: relative to this file
    repo_root = Path(__file__).resolve().parents[3]
    for candidate in (
        repo_root / "config" / "config.local.yaml",
        repo_root / "config" / "config.example.yaml",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No config file found. Set DR_CONFIG or create config/config.local.yaml")


def load_config(path: str | os.PathLike[str] | None = None) -> AppConfig:
    cfg_path = _resolve_path(path)
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    interpolated = _interpolate(raw)
    return AppConfig.model_validate(interpolated)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return load_config()
