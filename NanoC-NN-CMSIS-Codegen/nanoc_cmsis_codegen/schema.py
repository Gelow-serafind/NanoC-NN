from __future__ import annotations

from typing import Any

from .model import ValidationIssue

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

REQUIRED_TENSOR_FIELDS = {"name", "elem_type", "shape"}
REQUIRED_INITIALIZER_FIELDS = {"name", "elem_type", "shape", "c_name", "role"}
REQUIRED_NODE_FIELDS = {
    "index",
    "name",
    "op_type",
    "inputs",
    "outputs",
    "normalized_attributes",
    "input_shapes",
    "output_shapes",
    "weights",
    "status",
}


def missing_required_fields(model_graph: dict[str, Any]) -> list[str]:
    return sorted(REQUIRED_MODEL_GRAPH_FIELDS - set(model_graph))


def validate_model_graph(model_graph: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for field in missing_required_fields(model_graph):
        issues.append(ValidationIssue("error", field, "required field is missing"))

    if "schema_version" not in model_graph:
        issues.append(
            ValidationIssue(
                "warning",
                "schema_version",
                "converter output has no schema_version; treating it as legacy v0",
            )
        )
    if "quantization" not in model_graph:
        issues.append(
            ValidationIssue(
                "warning",
                "quantization",
                "converter output has no quantization section; CMSIS-NN int8 calls are blocked",
            )
        )

    _validate_list_field(model_graph, "inputs", REQUIRED_TENSOR_FIELDS, issues)
    _validate_list_field(model_graph, "outputs", REQUIRED_TENSOR_FIELDS, issues)
    _validate_list_field(model_graph, "initializers", REQUIRED_INITIALIZER_FIELDS, issues)
    _validate_list_field(model_graph, "nodes", REQUIRED_NODE_FIELDS, issues)

    layout = model_graph.get("layout")
    if layout not in {"NCHW", "NHWC"}:
        issues.append(
            ValidationIssue(
                "warning",
                "layout",
                f"layout should be explicit NCHW or NHWC, got {layout!r}",
            )
        )

    for tensor_role in ("inputs", "outputs"):
        for index, tensor in enumerate(model_graph.get(tensor_role, [])):
            if isinstance(tensor, dict):
                _validate_shape(
                    tensor.get("shape"),
                    f"{tensor_role}[{index}].shape",
                    issues,
                )

    for index, node in enumerate(model_graph.get("nodes", [])):
        if not isinstance(node, dict):
            continue
        for shape_key in ("input_shapes", "output_shapes"):
            shape_map = node.get(shape_key)
            if not isinstance(shape_map, dict):
                continue
            for name, shape in shape_map.items():
                _validate_shape(shape, f"nodes[{index}].{shape_key}.{name}", issues)

    return issues


def has_error(issues: list[ValidationIssue]) -> bool:
    return any(item.level == "error" for item in issues)


def _validate_list_field(
    model_graph: dict[str, Any],
    field: str,
    required_fields: set[str],
    issues: list[ValidationIssue],
) -> None:
    values = model_graph.get(field)
    if values is None:
        return
    if not isinstance(values, list):
        issues.append(ValidationIssue("error", field, "field must be a list"))
        return
    for index, item in enumerate(values):
        item_field = f"{field}[{index}]"
        if not isinstance(item, dict):
            issues.append(ValidationIssue("error", item_field, "entry must be an object"))
            continue
        for missing in sorted(required_fields - set(item)):
            issues.append(
                ValidationIssue(
                    "error",
                    f"{item_field}.{missing}",
                    "required field is missing",
                )
            )


def _validate_shape(
    shape: Any,
    field: str,
    issues: list[ValidationIssue],
) -> None:
    if not isinstance(shape, list):
        issues.append(ValidationIssue("error", field, "shape must be a list"))
        return
    for axis, dim in enumerate(shape):
        if dim is None or isinstance(dim, int | str):
            continue
        issues.append(
            ValidationIssue(
                "error",
                f"{field}[{axis}]",
                f"shape dimension must be int, string, or null, got {type(dim).__name__}",
            )
        )
