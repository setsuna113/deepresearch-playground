"""SearcherAgent — produces a query string for one sub-question, runs the
configured search provider, and appends SearchDocuments to ctx."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepresearch.agents._jsonparse import parse_json
from deepresearch.agents.base import Agent
from deepresearch.agents.prompts import load_prompt
from deepresearch.schemas.agents import AgentRole
from deepresearch.schemas.search import SearchDocument, SubQuestion

if TYPE_CHECKING:
    from deepresearch.agents.context import RunContext


class SearcherAgent(Agent):
    role = AgentRole.searcher

    def __init__(self, sq: SubQuestion) -> None:
        self.sq = sq

    async def _formulate_query(self, ctx: "RunContext") -> str:
        prompt = load_prompt("searcher").format(subquestion=self.sq.text)
        endpoint = ctx.deps.router.select(
            profile=ctx.request.model_profile, role="searcher", envelope=ctx.request.privacy_envelope
        )
        resp = await ctx.deps.model_client.complete(
            endpoint_name=endpoint.name,
            messages=[{"role": "user", "content": prompt}],
            role="searcher",
            run_id=ctx.run.id,
            envelope=ctx.request.privacy_envelope,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        try:
            data = parse_json(resp.text)
            return str(data.get("query") or self.sq.text)
        except Exception:
            return self.sq.text

    async def _run(self, ctx: "RunContext") -> dict:
        query = await self._formulate_query(ctx)
        provider = ctx.deps.tools.get_search(ctx.deps.config.search.default_provider)
        max_results = ctx.deps.config.search.providers.get(provider.name)
        n = max_results.max_results if max_results else 5
        hits = await provider.search(query, max_results=n)
        added: list[SearchDocument] = []
        for h in hits:
            doc = SearchDocument(
                subquestion_id=self.sq.id,
                url=h.url,
                title=h.title,
                snippet=h.snippet,
                source_provider=h.provider,
                score=h.score,
            )
            added.append(doc)
            ctx.documents.append(doc)
            ctx.deps.repos.docs.append(ctx.run.id, doc)
        ctx.run.metrics.n_searches += 1
        ctx.run.metrics.n_documents += len(added)
        return {
            "subquestion_id": self.sq.id,
            "query": query,
            "hits": len(added),
            "urls": [d.url for d in added],
        }
