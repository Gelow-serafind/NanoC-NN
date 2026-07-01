from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model import (
    CodegenError,
    CodegenInput,
    InitializerSpec,
    ModelGraph,
    NodeSpec,
    TensorSpec,
)
from .schema import has_error, validate_model_graph


def load_codegen_input(input_dir: Path) -> CodegenInput:
    graph_path = input_dir / "model_graph.json"
    if not input_dir.exists():
        raise CodegenError(f"input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise CodegenError(f"input path is not a directory: {input_dir}")
    if not graph_path.exists():
        raise CodegenError(f"model_graph.json does not exist in input directory: {input_dir}")

    try:
        raw_model_graph = json.loads(graph_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CodegenError(f"failed to parse model_graph.json: {exc}") from exc

    if not isinstance(raw_model_graph, dict):
        raise CodegenError("model_graph.json root must be an object")

    validation_issues = validate_model_graph(raw_model_graph)
    if has_error(validation_issues):
        errors = "; ".join(
            f"{item.field}: {item.message}"
            for item in validation_issues
            if item.level == "error"
        )
        raise CodegenError(f"invalid converter output schema: {errors}")

    return CodegenInput(
        model_graph=_parse_model_graph(raw_model_graph),
        raw_model_graph=raw_model_graph,
        input_dir=input_dir,
        validation_issues=validation_issues,
        conversion_report=_read_optional(input_dir / "conversion_report.txt"),
        weights_header=_read_optional(input_dir / "weights.h"),
    )


def _parse_model_graph(raw: dict[str, Any]) -> ModelGraph:
    return ModelGraph(
        model_path=str(raw.get("model_path", "")),
        ir_version=_optional_int(raw.get("ir_version")),
        opsets=_parse_opsets(raw.get("opsets", {})),
        producer_name=str(raw.get("producer_name", "")),
        producer_version=str(raw.get("producer_version", "")),
        graph_name=str(raw.get("graph_name", "")),
        layout=str(raw.get("layout", "NCHW")),
        inputs=[_parse_tensor(item) for item in raw.get("inputs", [])],
        outputs=[_parse_tensor(item) for item in raw.get("outputs", [])],
        initializers=[_parse_initializer(item) for item in raw.get("initializers", [])],
        nodes=[_parse_node(item) for item in raw.get("nodes", [])],
        warnings=[str(item) for item in raw.get("warnings", [])],
        errors=[str(item) for item in raw.get("errors", [])],
        unsupported_ops=[str(item) for item in raw.get("unsupported_ops", [])],
        c_exportable_initializers=[
            str(item) for item in raw.get("c_exportable_initializers", [])
        ],
        schema_version=(
            None if raw.get("schema_version") is None else str(raw.get("schema_version"))
        ),
        quantization=raw.get("quantization") if isinstance(raw.get("quantization"), dict) else None,
    )


def _parse_tensor(raw: dict[str, Any]) -> TensorSpec:
    return TensorSpec(
        name=str(raw.get("name", "")),
        elem_type=str(raw.get("elem_type", "UNKNOWN")),
        shape=_parse_shape(raw.get("shape", [])),
        raw_shape=_parse_shape(raw.get("raw_shape", raw.get("shape", []))),
        dynamic_axes=[int(item) for item in raw.get("dynamic_axes", [])],
        is_dynamic=bool(raw.get("is_dynamic", False)),
    )


def _parse_initializer(raw: dict[str, Any]) -> InitializerSpec:
    return InitializerSpec(
        name=str(raw.get("name", "")),
        elem_type=str(raw.get("elem_type", "UNKNOWN")),
        shape=_parse_shape(raw.get("shape", [])),
        element_count=_optional_int(raw.get("element_count")),
        c_name=str(raw.get("c_name", "")),
        role=str(raw.get("role", "")),
        is_float32=bool(raw.get("is_float32", False)),
        is_c_exportable=bool(raw.get("is_c_exportable", False)),
    )


def _parse_node(raw: dict[str, Any]) -> NodeSpec:
    return NodeSpec(
        index=int(raw.get("index", 0)),
        name=str(raw.get("name", "")),
        op_type=str(raw.get("op_type", "")),
        inputs=[str(item) for item in raw.get("inputs", [])],
        outputs=[str(item) for item in raw.get("outputs", [])],
        attributes=dict(raw.get("attributes", {})),
        normalized_attributes=dict(raw.get("normalized_attributes", {})),
        input_shapes=_parse_shape_map(raw.get("input_shapes", {})),
        output_shapes=_parse_shape_map(raw.get("output_shapes", {})),
        weights=[str(item) for item in raw.get("weights", [])],
        converter_status=str(raw.get("status", "unknown")),
    )


def _parse_opsets(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    result: dict[str, int] = {}
    for key, value in raw.items():
        parsed = _optional_int(value)
        if parsed is not None:
            result[str(key)] = parsed
    return result


def _parse_shape_map(raw: Any) -> dict[str, list[int | str | None]]:
    if not isinstance(raw, dict):
        return {}
    return {str(name): _parse_shape(shape) for name, shape in raw.items()}


def _parse_shape(raw: Any) -> list[int | str | None]:
    if not isinstance(raw, list):
        return []
    result: list[int | str | None] = []
    for item in raw:
        if item is None or isinstance(item, int | str):
            result.append(item)
        else:
            result.append(str(item))
    return result


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_optional(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")
