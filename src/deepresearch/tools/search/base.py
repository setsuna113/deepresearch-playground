"""Search provider protocol + the common SearchHit shape."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class SearchHit(BaseModel):
    url: str
    title: str | None = None
    snippet: str | None = None
    score: float | None = None
    provider: str


class SearchProvider(Protocol):
    name: str

    async def search(self, query: str, *, max_results: int) -> list[SearchHit]: ...
