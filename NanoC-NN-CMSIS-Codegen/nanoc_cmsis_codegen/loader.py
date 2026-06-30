from __future__ import annotations

import json
from pathlib import Path

from .model import CodegenInput


def load_codegen_input(input_dir: Path) -> CodegenInput:
    graph_path = input_dir / "model_graph.json"
    model_graph = json.loads(graph_path.read_text(encoding="utf-8"))
    return CodegenInput(model_graph=model_graph, input_dir=input_dir)
