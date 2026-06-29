from __future__ import annotations

from pathlib import Path
from typing import Any

import onnx
from onnx import AttributeProto, GraphProto, TensorProto, checker, helper, numpy_helper

from .model import (
    ConversionError,
    ConversionOptions,
    InitializerInfo,
    ModelInfo,
    NodeInfo,
    TensorInfo,
)
from .naming import make_unique_c_names
from .shape import tensor_dtype_name, tensor_info_from_value_info

SUPPORTED_OPS = {
    "Add",
    "BatchNormalization",
    "Cast",
    "Constant",
    "Conv",
    "Flatten",
    "Gemm",
    "GlobalAveragePool",
    "MatMul",
    "MaxPool",
    "Relu",
    "Reshape",
    "Softmax",
    "Transpose",
}
SUPPORTED_OPSET_MIN = 11
SUPPORTED_OPSET_MAX = 17

PARAMETER_INITIALIZER_INPUTS = {
    "Add": {1},
    "BatchNormalization": {1, 2, 3, 4},
    "Conv": {1, 2},
    "Gemm": {1, 2},
    "MatMul": {1},
}
AUXILIARY_INITIALIZER_INPUTS = {
    "Reshape": {1},
}


def parse_model(options: ConversionOptions) -> ModelInfo:
    model_path = Path(options.model_path)
    warnings: list[str] = []
    errors: list[str] = []

    if not model_path.exists():
        raise ConversionError(f"model file does not exist: {model_path}")
    if not model_path.is_file():
        raise ConversionError(f"model path is not a file: {model_path}")

    try:
        model = onnx.load(model_path)
        checker.check_model(model)
    except Exception as exc:  # noqa: BLE001 - surface ONNX checker detail to CLI.
        raise ConversionError(f"failed to load or check ONNX model: {exc}") from exc

    try:
        inferred_model = onnx.shape_inference.infer_shapes(model)
    except Exception as exc:  # noqa: BLE001 - keep conversion useful with partial shapes.
        warnings.append(f"ONNX shape inference failed; continuing with original shapes: {exc}")
        inferred_model = model

    opsets = _opset_imports(model)
    default_opset = opsets.get("", opsets.get("ai.onnx"))
    if default_opset is not None and not (
        SUPPORTED_OPSET_MIN <= default_opset <= SUPPORTED_OPSET_MAX
    ):
        warnings.append(
            f"ONNX opset {default_opset} is outside initial support range "
            f"{SUPPORTED_OPSET_MIN}-{SUPPORTED_OPSET_MAX}."
        )

    graph = inferred_model.graph
    initializer_names = [initializer.name for initializer in graph.initializer]
    c_names = make_unique_c_names(initializer_names, prefix=options.prefix)
    initializer_name_set = set(initializer_names)
    initializer_roles = _classify_initializer_roles(graph, initializer_name_set)
    initializers = _parse_initializers(
        graph,
        c_names,
        initializer_roles,
        warnings,
        options.strict,
    )

    value_info_by_name = _collect_value_info(graph, options, warnings)
    inputs = [
        value_info_by_name[value.name]
        for value in graph.input
        if value.name not in initializer_name_set and value.name in value_info_by_name
    ]
    outputs = [
        value_info_by_name[value.name]
        for value in graph.output
        if value.name in value_info_by_name
    ]

    nodes = _parse_nodes(graph, value_info_by_name, initializer_name_set, initializers, warnings)

    unsupported_ops = sorted({node.op_type for node in nodes if node.status == "unsupported"})
    if unsupported_ops:
        message = f"unsupported ONNX ops found: {', '.join(unsupported_ops)}"
        if options.strict:
            raise ConversionError(message)
        warnings.append(message)

    return ModelInfo(
        model_path=model_path,
        ir_version=model.ir_version,
        opsets=opsets,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        graph_name=graph.name,
        layout=options.layout,
        inputs=inputs,
        outputs=outputs,
        initializers=initializers,
        nodes=nodes,
        warnings=warnings,
        errors=errors,
    )


def _opset_imports(model: onnx.ModelProto) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in model.opset_import:
        result[item.domain] = int(item.version)
    return result


def _parse_initializers(
    graph: GraphProto,
    c_names: dict[str, str],
    roles: dict[str, str],
    warnings: list[str],
    strict: bool,
) -> list[InitializerInfo]:
    result: list[InitializerInfo] = []
    for initializer in graph.initializer:
        array = numpy_helper.to_array(initializer)
        elem_type = tensor_dtype_name(initializer.data_type)
        role = roles.get(initializer.name, "parameter")
        if elem_type != "FLOAT" and role == "parameter":
            message = (
                f"initializer '{initializer.name}' has unsupported data type {elem_type}; "
                "only float32 weights are exported."
            )
            if strict:
                raise ConversionError(message)
            warnings.append(message)
        result.append(
            InitializerInfo(
                name=initializer.name,
                elem_type=elem_type,
                shape=[int(item) for item in array.shape],
                element_count=int(array.size),
                c_name=c_names[initializer.name],
                role=role,
                array=array,
            )
        )
    return result


def _classify_initializer_roles(
    graph: GraphProto,
    initializer_names: set[str],
) -> dict[str, str]:
    roles = {name: "unused_constant" for name in initializer_names}
    for node in graph.node:
        for input_index, input_name in enumerate(node.input):
            if input_name not in initializer_names:
                continue
            if input_index in PARAMETER_INITIALIZER_INPUTS.get(node.op_type, set()):
                roles[input_name] = "parameter"
            elif input_index in AUXILIARY_INITIALIZER_INPUTS.get(node.op_type, set()):
                roles[input_name] = _merge_initializer_role(
                    roles[input_name],
                    "auxiliary_constant",
                )
            else:
                roles[input_name] = _merge_initializer_role(roles[input_name], "parameter")
    return roles


def _merge_initializer_role(current: str, new: str) -> str:
    priority = {
        "unused_constant": 0,
        "auxiliary_constant": 1,
        "parameter": 2,
    }
    return new if priority[new] > priority[current] else current


def _collect_value_info(
    graph: GraphProto,
    options: ConversionOptions,
    warnings: list[str],
) -> dict[str, TensorInfo]:
    result: dict[str, TensorInfo] = {}
    value_infos = list(graph.input) + list(graph.value_info) + list(graph.output)
    for value_info in value_infos:
        if not value_info.type.HasField("tensor_type"):
            warnings.append(f"value '{value_info.name}' is not a tensor; shape is skipped.")
            continue
        result[value_info.name] = tensor_info_from_value_info(
            value_info,
            batch_size=options.batch_size,
            strict=options.strict,
            warnings=warnings,
            role="value",
        )
    return result


def _parse_nodes(
    graph: GraphProto,
    value_info_by_name: dict[str, TensorInfo],
    initializer_name_set: set[str],
    initializers: list[InitializerInfo],
    warnings: list[str],
) -> list[NodeInfo]:
    initializer_by_name = {item.name: item for item in initializers}
    nodes: list[NodeInfo] = []
    for index, node in enumerate(graph.node):
        attributes = extract_attributes(node.attribute)
        weights = [
            name
            for name in node.input
            if name in initializer_name_set and initializer_by_name[name].role == "parameter"
        ]
        normalized = normalize_attributes(
            op_type=node.op_type,
            attributes=attributes,
            weight_infos=[initializer_by_name[name] for name in weights],
        )
        status = "supported" if node.op_type in SUPPORTED_OPS else "unsupported"
        if status == "unsupported":
            warnings.append(f"node {index} uses unsupported op '{node.op_type}'.")

        nodes.append(
            NodeInfo(
                index=index,
                name=node.name or f"{node.op_type}_{index}",
                op_type=node.op_type,
                inputs=list(node.input),
                outputs=list(node.output),
                attributes=attributes,
                normalized_attributes=normalized,
                input_shapes=_shapes_for_names(
                    node.input,
                    value_info_by_name,
                    initializer_by_name,
                ),
                output_shapes=_shapes_for_names(node.output, value_info_by_name),
                weights=weights,
                status=status,
            )
        )
    return nodes


def _shapes_for_names(
    names: list[str] | tuple[str, ...],
    value_info_by_name: dict[str, TensorInfo],
    initializer_by_name: dict[str, InitializerInfo] | None = None,
) -> dict[str, list[int | str | None]]:
    shape_map: dict[str, list[int | str | None]] = {}
    initializers = initializer_by_name or {}
    for name in names:
        if name in value_info_by_name:
            shape_map[name] = value_info_by_name[name].shape
        elif name in initializers:
            shape_map[name] = initializers[name].shape
    return shape_map


def extract_attributes(attributes: list[AttributeProto]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for attribute in attributes:
        result[attribute.name] = _plain_attribute_value(helper.get_attribute_value(attribute))
    return result


def _plain_attribute_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, TensorProto):
        array = numpy_helper.to_array(value)
        values = _plain_attribute_value(array.tolist())
        return {
            "name": value.name,
            "data_type": tensor_dtype_name(value.data_type),
            "dims": [int(item) for item in value.dims],
            "element_count": int(array.size),
            "values": values if int(array.size) <= 64 else None,
            "preview": values[:16] if isinstance(values, list) and int(array.size) > 64 else None,
        }
    if isinstance(value, GraphProto):
        return {"name": value.name, "node_count": len(value.node)}
    if isinstance(value, tuple | list):
        return [_plain_attribute_value(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def normalize_attributes(
    *,
    op_type: str,
    attributes: dict[str, Any],
    weight_infos: list[InitializerInfo],
) -> dict[str, Any]:
    if op_type == "Conv":
        weight_shape = weight_infos[0].shape if weight_infos else []
        spatial_rank = max(len(weight_shape) - 2, 0)
        kernel_shape = _as_int_list(attributes.get("kernel_shape"))
        if not kernel_shape and spatial_rank:
            kernel_shape = weight_shape[2:]
        spatial_rank = len(kernel_shape) or spatial_rank or 2
        return {
            "kernel_shape": kernel_shape,
            "strides": _as_int_list(attributes.get("strides"), [1] * spatial_rank),
            "pads": _as_int_list(attributes.get("pads"), [0] * (spatial_rank * 2)),
            "dilations": _as_int_list(attributes.get("dilations"), [1] * spatial_rank),
            "group": int(attributes.get("group", 1)),
        }
    if op_type == "MaxPool":
        kernel_shape = _as_int_list(attributes.get("kernel_shape"))
        spatial_rank = len(kernel_shape) or 2
        return {
            "kernel_shape": kernel_shape,
            "strides": _as_int_list(attributes.get("strides"), kernel_shape or [1] * spatial_rank),
            "pads": _as_int_list(attributes.get("pads"), [0] * (spatial_rank * 2)),
        }
    if op_type == "Gemm":
        return {
            "alpha": float(attributes.get("alpha", 1.0)),
            "beta": float(attributes.get("beta", 1.0)),
            "transA": int(attributes.get("transA", 0)),
            "transB": int(attributes.get("transB", 0)),
        }
    if op_type == "Flatten":
        return {"axis": int(attributes.get("axis", 1))}
    if op_type == "Add":
        return {
            "broadcast": "numpy",
        }
    if op_type == "Constant":
        return attributes
    if op_type == "Transpose":
        perm = _as_int_list(attributes.get("perm"))
        return {
            "perm": perm,
            "uses_default_reverse_perm": not perm,
        }
    if op_type == "Cast":
        to_type = int(attributes.get("to", 0))
        return {
            "to": to_type,
            "to_dtype": tensor_dtype_name(to_type),
            "saturate": int(attributes.get("saturate", 1)),
        }
    if op_type == "BatchNormalization":
        return {
            "epsilon": float(attributes.get("epsilon", 1e-5)),
            "momentum": float(attributes.get("momentum", 0.9)),
            "training_mode": int(attributes.get("training_mode", 0)),
        }
    if op_type == "GlobalAveragePool":
        return {
            "spatial_axes": "all_axes_after_batch_and_channel",
        }
    if op_type == "Reshape":
        return {
            "allowzero": int(attributes.get("allowzero", 0)),
        }
    return attributes


def _as_int_list(value: Any, default: list[int] | None = None) -> list[int]:
    if value is None:
        return list(default or [])
    if isinstance(value, int):
        return [value]
    return [int(item) for item in value]
