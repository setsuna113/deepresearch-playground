"""Tool registry — used by orchestrator/agents to look up providers by name.

Phase 1 is small (one search provider + one fetcher), but the registry is
already a seam: Phase 2 will add browser-based fetchers and a Verifier agent
that uses an entirely different tool surface.
"""

from __future__ import annotations

from dataclasses import dataclass

from deepresearch.tools.fetch.fetcher import PageFetcher
from deepresearch.tools.search.base import SearchProvider


@dataclass
class ToolRegistry:
    search: dict[str, SearchProvider]
    fetcher: PageFetcher

    def get_search(self, name: str) -> SearchProvider:
        try:
            return self.search[name]
        except KeyError as e:
            raise KeyError(f"unknown search provider '{name}' (have: {sorted(self.search)})") from e
