"""ReflectorAgent — emits structured ReflectionUpdate JSON describing
what should be written to memory after this run."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepresearch.agents._jsonparse import parse_json
from deepresearch.agents.base import Agent
from deepresearch.agents.prompts import load_prompt
from deepresearch.schemas.agents import AgentRole, ReflectionUpdate

if TYPE_CHECKING:
    from deepresearch.agents.context import RunContext


class ReflectorAgent(Agent):
    role = AgentRole.reflector

    async def _run(self, ctx: "RunContext") -> dict:
        plan_text = "\n".join(f"- {sq.text}" for sq in ctx.plan) or "(no plan)"
        prompt = load_prompt("reflector").format(
            query=ctx.request.query,
            plan=plan_text,
            report=ctx.draft_report or "(no report)",
        )
        endpoint = ctx.deps.router.select(
            profile=ctx.request.model_profile, role="reflector",
            envelope=ctx.request.privacy_envelope,
        )
        resp = await ctx.deps.model_client.complete(
            endpoint_name=endpoint.name,
            messages=[{"role": "user", "content": prompt}],
            role="reflector",
            run_id=ctx.run.id,
            envelope=ctx.request.privacy_envelope,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        try:
            data = parse_json(resp.text)
            update = ReflectionUpdate.model_validate(data)
        except Exception as e:
            update = ReflectionUpdate(needs_revision=False)
            return {"error": repr(e), "raw": resp.text, "model_call_id": str(resp.call_id)}
        ctx.reflection = update
        return {
            "personal_update": update.personal_update,
            "procedural_update": update.procedural_update,
            "tool_update": update.tool_update,
            "needs_revision": update.needs_revision,
            "model_call_id": str(resp.call_id),
        }
