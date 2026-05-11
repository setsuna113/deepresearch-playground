"""SynthesizerAgent — produces the final Markdown report and citations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepresearch.agents._jsonparse import parse_json
from deepresearch.agents.base import Agent
from deepresearch.agents.prompts import load_prompt
from deepresearch.schemas.agents import AgentRole
from deepresearch.schemas.search import Citation

if TYPE_CHECKING:
    from deepresearch.agents.context import RunContext


class SynthesizerAgent(Agent):
    role = AgentRole.synthesizer

    async def _run(self, ctx: "RunContext") -> dict:
        ev_lines: list[str] = []
        for i, ev in enumerate(ctx.evidence, start=1):
            ev_lines.append(f"[{i}] ({ev.url}) {ev.quote}")
        evidence_block = "\n".join(ev_lines) if ev_lines else "(no evidence gathered)"
        sq_lines = "\n".join(f"- {sq.text}" for sq in ctx.plan)

        prompt = load_prompt("synthesizer").format(
            query=ctx.request.query,
            subquestions=sq_lines or "(none)",
            evidence_block=evidence_block,
        )
        endpoint = ctx.deps.router.select(
            profile=ctx.request.model_profile, role="synthesizer",
            envelope=ctx.request.privacy_envelope,
        )
        resp = await ctx.deps.model_client.complete(
            endpoint_name=endpoint.name,
            messages=[{"role": "user", "content": prompt}],
            role="synthesizer",
            run_id=ctx.run.id,
            envelope=ctx.request.privacy_envelope,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        report_md = ""
        citations: list[Citation] = []
        try:
            data = parse_json(resp.text)
            report_md = str(data.get("report_md", "")).strip()
            for c in data.get("citations", []):
                citations.append(
                    Citation(
                        marker=str(c.get("marker", "")),
                        url=str(c.get("url", "")),
                        title=c.get("title"),
                        quote=c.get("quote"),
                    )
                )
        except Exception as e:
            report_md = f"(synthesizer parse error: {e!r})\n\nraw:\n{resp.text}"
        ctx.draft_report = report_md
        ctx.citations = citations
        return {
            "report_md_chars": len(report_md),
            "n_citations": len(citations),
            "model_call_id": str(resp.call_id),
        }
