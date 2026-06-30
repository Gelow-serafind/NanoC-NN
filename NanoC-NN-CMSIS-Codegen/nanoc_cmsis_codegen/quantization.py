from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantizationRequirement:
    node_name: str
    requirement: str
    status: str = "blocked"


def no_quantization_requirements() -> list[QuantizationRequirement]:
    return []
