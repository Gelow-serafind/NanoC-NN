from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BufferPlan:
    name: str
    size_bytes: int
    reason: str


def empty_memory_plan() -> list[BufferPlan]:
    return []
