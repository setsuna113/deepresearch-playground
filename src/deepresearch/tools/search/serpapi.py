"""SerpAPI stub — kept thin so Phase 2 can swap providers freely."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from deepresearch.tools.search.base import SearchHit


@dataclass
class SerpAPISearch:
    api_key: str
    name: str = "serpapi"
    timeout_s: float = 20.0

    async def search(self, query: str, *, max_results: int = 8) -> list[SearchHit]:
        if not self.api_key:
            return []
        params = {
            "api_key": self.api_key,
            "q": query,
            "engine": "google",
            "num": max_results,
        }
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.get("https://serpapi.com/search", params=params)
            resp.raise_for_status()
            data = resp.json()
        out: list[SearchHit] = []
        for r in data.get("organic_results", [])[:max_results]:
            out.append(
                SearchHit(
                    url=r.get("link", ""),
                    title=r.get("title"),
                    snippet=r.get("snippet"),
                    provider=self.name,
                )
            )
        return out
