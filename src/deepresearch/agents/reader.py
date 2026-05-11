"""ReaderAgent — fetches docs for one sub-question and extracts short
evidence quotes via the LLM."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepresearch.agents._jsonparse import parse_json
from deepresearch.agents.base import Agent
from deepresearch.agents.prompts import load_prompt
from deepresearch.schemas.agents import AgentRole
from deepresearch.schemas.search import Evidence, SubQuestion
from deepresearch.tools.fetch.extract import extract_content

if TYPE_CHECKING:
    from deepresearch.agents.context import RunContext


_MAX_PAGE_CHARS = 8000


class ReaderAgent(Agent):
    role = AgentRole.reader

    def __init__(self, sq: SubQuestion) -> None:
        self.sq = sq

    async def _run(self, ctx: "RunContext") -> dict:
        docs = [d for d in ctx.documents if d.subquestion_id == self.sq.id]
        added_evidence = 0
        fetched_urls: list[str] = []
        for doc in docs[:3]:  # cap docs per subquestion
            result = await ctx.deps.tools.fetcher.fetch(doc.url)
            if not result.html:
                continue
            ext = extract_content(result.html, url=doc.url)
            if not ext.text:
                continue
            doc.content_md = ext.text[:_MAX_PAGE_CHARS]
            if ext.title and not doc.title:
                doc.title = ext.title
            fetched_urls.append(doc.url)

            prompt = load_prompt("reader").format(
                subquestion=self.sq.text, url=doc.url, content=doc.content_md
            )
            endpoint = ctx.deps.router.select(
                profile=ctx.request.model_profile, role="reader",
                envelope=ctx.request.privacy_envelope,
            )
            resp = await ctx.deps.model_client.complete(
                endpoint_name=endpoint.name,
                messages=[{"role": "user", "content": prompt}],
                role="reader",
                run_id=ctx.run.id,
                envelope=ctx.request.privacy_envelope,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            try:
                data = parse_json(resp.text)
                for e in data.get("evidence", []):
                    quote = str(e.get("quote", "")).strip()
                    if not quote:
                        continue
                    ctx.evidence.append(
                        Evidence(
                            document_id=doc.id,
                            subquestion_id=self.sq.id,
                            quote=quote,
                            url=doc.url,
                            relevance=float(e.get("relevance", 0.0) or 0.0),
                        )
                    )
                    added_evidence += 1
            except Exception:
                continue
        return {
            "subquestion_id": self.sq.id,
            "fetched": len(fetched_urls),
            "evidence_added": added_evidence,
        }
