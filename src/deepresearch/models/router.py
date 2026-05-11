"""Router — picks an endpoint for a (profile, role, envelope, hint).

Phase 1: pure profile lookup; `envelope` and `hint` are accepted but unused.
Phase 4 (ParetoDispatch) replaces the body of `select()` to consider
privacy leakage, GPU utilization, deadline, and the privacy budget — the
signature is intentionally future-proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deepresearch.models.endpoints import EndpointSet
from deepresearch.schemas.models import Endpoint
from deepresearch.schemas.privacy import PrivacyEnvelope


@dataclass
class SchedulingHint:
    """Optional hint passed by callers; ignored in Phase 1.

    In Phase 4 ParetoDispatch reads these to pick a Pareto-optimal endpoint.
    """

    deadline_ms: int | None = None
    expected_cost_tokens: int | None = None
    privacy_budget_bits: float | None = None
    extra: dict[str, Any] | None = None


class Router:
    def __init__(self, endpoints: EndpointSet) -> None:
        self._endpoints = endpoints

    def select(
        self,
        *,
        profile: str,
        role: str,
        envelope: PrivacyEnvelope | None = None,
        hint: SchedulingHint | None = None,
    ) -> Endpoint:
        # Phase 1: profile-only. Envelope + hint are reserved.
        _ = envelope, hint
        return self._endpoints.endpoint_for(profile=profile, role=role)
