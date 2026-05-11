"""4-type ↔ ReMe-3-type mapping.

Our agent layer speaks the 4-type vocabulary:
    personal | procedural | tool | working

ReMe natively supports 3 of these. We own `working` ourselves in a separate
Qdrant collection.

| Our type    | Backend           |
| ----------- | ----------------- |
| personal    | ReMe `personal`   |
| procedural  | ReMe `procedural` |
| tool        | ReMe `tool`       |
| working     | working_qdrant    |
"""

from __future__ import annotations

from deepresearch.schemas.memory import MemoryType

# Names ReMe expects on the wire. Keep in one place so the adapter can be
# patched if ReMe's spelling drifts.
REME_PERSONAL = "personal"
REME_PROCEDURAL = "procedural"
REME_TOOL = "tool"

_REME_MAP = {
    MemoryType.personal: REME_PERSONAL,
    MemoryType.procedural: REME_PROCEDURAL,
    MemoryType.tool: REME_TOOL,
}


def reme_type_for(t: MemoryType) -> str | None:
    """Return the ReMe-native type name, or None if this type isn't ReMe-backed."""
    return _REME_MAP.get(t)


def working_qualifies(t: MemoryType) -> bool:
    return t == MemoryType.working
