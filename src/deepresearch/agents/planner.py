"""PlannerAgent — decompose the query into sub-questions, biased by prior memory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepresearch.agents._jsonparse import parse_json
from deepresearch.agents.base import Agent
from deepresearch.agents.prompts import load_prompt
from deepresearch.schemas.agents import AgentRole
from deepresearch.schemas.memory import MemoryType
from deepresearch.schemas.search import SubQuestion

if TYPE_CHECKING:
    from deepresearch.agents.context import RunContext


def _format_records(records: list, default: str = "(none)") -> str:
    if not records:
        return default
    return "\n".join(f"- {r.content}" for r in records[:5])


class PlannerAgent(Agent):
    role = AgentRole.planner

    async def _run(self, ctx: "RunContext") -> dict:
        prompt = load_prompt("planner").format(
            query=ctx.request.query,
            personal=_format_records(ctx.primes.get(MemoryType.personal, [])),
            procedural=_format_records(ctx.primes.get(MemoryType.procedural, [])),
            tool=_format_records(ctx.primes.get(MemoryType.tool, [])),
        )
        endpoint = ctx.deps.router.select(
            profile=ctx.request.model_profile, role="planner", envelope=ctx.request.privacy_envelope
        )
        resp = await ctx.deps.model_client.complete(
            endpoint_name=endpoint.name,
            messages=[{"role": "user", "content": prompt}],
            role="planner",
            run_id=ctx.run.id,
            envelope=ctx.request.privacy_envelope,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        try:
            data = parse_json(resp.text)
            raw = data.get("subquestions", [])
            ctx.plan = [
                SubQuestion(
                    id=str(sq.get("id") or f"sq{i + 1}"),
                    text=str(sq.get("text", "")),
                    rationale=sq.get("rationale"),
                )
                for i, sq in enumerate(raw)
                if sq.get("text")
            ]
        except Exception as e:
            ctx.plan = []
            return {"error": repr(e), "raw": resp.text, "n_subquestions": 0,
                    "model_call_id": str(resp.call_id)}
        return {"subquestions": [s.model_dump() for s in ctx.plan],
                "n_subquestions": len(ctx.plan),
                "model_call_id": str(resp.call_id)}
