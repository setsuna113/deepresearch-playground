"""MemoryProfile — controls per-type top_k and score thresholds."""

from __future__ import annotations

from dataclasses import dataclass

from deepresearch.config.schema import MemoryProfileConfig
from deepresearch.schemas.memory import MemoryType


@dataclass
class MemoryProfile:
    name: str
    personal_top_k: int
    procedural_top_k: int
    tool_top_k: int
    working_top_k: int
    score_floor: float

    @classmethod
    def from_config(cls, name: str, cfg: MemoryProfileConfig) -> MemoryProfile:
        return cls(
            name=name,
            personal_top_k=cfg.personal_top_k,
            procedural_top_k=cfg.procedural_top_k,
            tool_top_k=cfg.tool_top_k,
            working_top_k=cfg.working_top_k,
            score_floor=cfg.score_floor,
        )

    def top_k_for(self, t: MemoryType) -> int:
        return {
            MemoryType.personal: self.personal_top_k,
            MemoryType.procedural: self.procedural_top_k,
            MemoryType.tool: self.tool_top_k,
            MemoryType.working: self.working_top_k,
        }[t]
