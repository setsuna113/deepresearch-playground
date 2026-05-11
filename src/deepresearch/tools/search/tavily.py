"""Tavily search provider — async, OpenAI-compatible in the loose sense."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import structlog

from deepresearch.tools.search.base import SearchHit

log = structlog.get_logger(__name__)


@dataclass
class TavilySearch:
    api_key: str
    name: str = "tavily"
    timeout_s: float = 20.0

    async def search(self, query: str, *, max_results: int = 8) -> list[SearchHit]:
        if not self.api_key:
            log.warning("tavily_no_api_key")
            return []
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "advanced",
            "include_answer": False,
            "include_raw_content": False,
        }
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post("https://api.tavily.com/search", json=payload)
            resp.raise_for_status()
            data = resp.json()
        hits: list[SearchHit] = []
        for r in data.get("results", []):
            hits.append(
                SearchHit(
                    url=r.get("url", ""),
                    title=r.get("title"),
                    snippet=r.get("content") or r.get("snippet"),
                    score=r.get("score"),
                    provider=self.name,
                )
            )
        return hits
