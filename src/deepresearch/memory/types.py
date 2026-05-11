"""4-type ↔ ReMe-3-type mapping.

Our agent layer speaks the 4-type vocabulary:
    personal | task | tool | working

ReMe natively supports the first 3 of these (verified against reme-ai
0.3.1.8 source: task / tool / personal). We own `working` ourselves in a
separate Qdrant collection.

| Our type | Backend         |
| -------- | --------------- |
| personal | ReMe `personal` |
| task     | ReMe `task`     |
| tool     | ReMe `tool`     |
| working  | working_qdrant  |
"""

from __future__ import annotations

from deepresearch.schemas.memory import MemoryType

# Names ReMe expects on the wire. Keep in one place so the adapter can be
# patched if ReMe's spelling drifts.
REME_PERSONAL = "personal"
REME_TASK = "task"
REME_TOOL = "tool"

_REME_MAP = {
    MemoryType.personal: REME_PERSONAL,
    MemoryType.task: REME_TASK,
    MemoryType.tool: REME_TOOL,
}


def reme_type_for(t: MemoryType) -> str | None:
    """Return the ReMe-native type name, or None if this type isn't ReMe-backed."""
    return _REME_MAP.get(t)


def working_qualifies(t: MemoryType) -> bool:
    return t == MemoryType.working
