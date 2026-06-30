from __future__ import annotations

from typing import Any

REQUIRED_MODEL_GRAPH_FIELDS = {
    "model_path",
    "ir_version",
    "opsets",
    "layout",
    "inputs",
    "outputs",
    "initializers",
    "nodes",
    "warnings",
}


def missing_required_fields(model_graph: dict[str, Any]) -> list[str]:
    return sorted(REQUIRED_MODEL_GRAPH_FIELDS - set(model_graph))
