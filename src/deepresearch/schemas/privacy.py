"""PrivacyEnvelope — attached to every message, memory, doc, and model call.

Phase 1 defaults to {level: public, source: unknown}. Phase 4 fills in the
contextual-integrity (CI) fields and uses them to drive ParetoDispatch.
The shape is locked in now so later phases don't have to migrate data.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class PrivacyLevel(str, Enum):
    public = "public"
    internal = "internal"
    sensitive = "sensitive"
    secret = "secret"


class PrivacyEnvelope(BaseModel):
    """Contextual-integrity envelope.

    Phase 1 only uses `level` and `source`. The CI fields (subject, sender,
    recipient, attribute, transmission_principle) are reserved for Phase 4
    and may stay None until then.
    """

    level: PrivacyLevel = PrivacyLevel.public
    source: str = "unknown"
    subject: str | None = None
    sender: str | None = None
    recipient: str | None = None
    attribute: str | None = None
    transmission_principle: str | None = None
    sensitivity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)

    @classmethod
    def default_public(cls) -> PrivacyEnvelope:
        return cls(level=PrivacyLevel.public, source="unknown")
