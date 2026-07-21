from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
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
    "AveragePool",
    "BatchNormalization",
    "Cast",
    "Constant",
    "Conv",
    "Flatten",
    "Gemm",
    "Gather",
    "GlobalAveragePool",
    "MatMul",
    "MaxPool",
    "Concat",
    "Mul",
    "QLinearAdd",
    "QLinearConv",
    "QLinearGlobalAveragePool",
    "QLinearMatMul",
    "DequantizeLinear",
    "Dropout",
    "QuantizeLinear",
    "Relu",
    "Reshape",
    "Shape",
    "Slice",
    "Softmax",
    "Transpose",
    "Unsqueeze",
}
SUPPORTED_OPSET_MIN = 11
SUPPORTED_OPSET_MAX = 17

PARAMETER_INITIALIZER_INPUTS = {
    "Add": {1},
    "BatchNormalization": {1, 2, 3, 4},
    "Conv": {1, 2},
    "DequantizeLinear": {0},
    "Gemm": {1, 2},
    "MatMul": {1},
    "QLinearAdd": {3},
    "QLinearConv": {3, 8},
    "QLinearMatMul": {3},
    "QuantizeLinear": {0},
}
AUXILIARY_INITIALIZER_INPUTS = {
    "DequantizeLinear": {1, 2},
    "QLinearAdd": {1, 2, 4, 5, 6, 7},
    "QLinearConv": {1, 2, 4, 5, 6, 7},
    "QLinearGlobalAveragePool": {1, 2, 3, 4},
    "QLinearMatMul": {1, 2, 4, 5, 6, 7},
    "QuantizeLinear": {1, 2},
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
    _backfill_node_shapes(nodes)
    quantization = _extract_quantization(graph, initializers, nodes, warnings)

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
        quantization=quantization,
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
        if elem_type != "FLOAT" and role == "parameter" and elem_type not in {"INT8", "UINT8"}:
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


def _backfill_node_shapes(nodes: list[NodeInfo]) -> None:
    shapes: dict[str, list[int | str | None]] = {}
    for node in nodes:
        for name, shape in node.input_shapes.items():
            if shape:
                shapes.setdefault(name, shape)
        for name, shape in node.output_shapes.items():
            if shape:
                shapes.setdefault(name, shape)
    changed = True
    while changed:
        changed = False
        for node in nodes:
            for name in node.inputs:
                if name not in node.input_shapes and name in shapes:
                    node.input_shapes[name] = shapes[name]
                    changed = True
            for name in node.outputs:
                if name not in node.output_shapes and name in shapes:
                    node.output_shapes[name] = shapes[name]
                    changed = True
            inferred_outputs = _infer_node_output_shapes(node)
            for name, shape in inferred_outputs.items():
                if name in node.output_shapes or not shape:
                    continue
                node.output_shapes[name] = shape
                shapes[name] = shape
                changed = True
            for name, shape in node.output_shapes.items():
                if shape and name not in shapes:
                    shapes[name] = shape
                    changed = True


def _infer_node_output_shapes(node: NodeInfo) -> dict[str, list[int | str | None]]:
    if not node.outputs:
        return {}
    if node.op_type in {"Conv", "QLinearConv"}:
        shape = _infer_conv_output_shape(node)
        return {node.outputs[0]: shape} if shape else {}
    if node.op_type in {"GlobalAveragePool", "QLinearGlobalAveragePool"}:
        input_shape = _first_input_shape(node)
        if input_shape and len(input_shape) == 4:
            return {node.outputs[0]: [input_shape[0], input_shape[1], 1, 1]}
    if node.op_type == "Concat":
        shape = _infer_concat_output_shape(node)
        return {node.outputs[0]: shape} if shape else {}
    if node.op_type in {
        "Add",
        "QLinearAdd",
        "Relu",
        "Clip",
        "Dropout",
        "QuantizeLinear",
        "DequantizeLinear",
    }:
        input_shape = _first_input_shape(node)
        return {node.outputs[0]: list(input_shape)} if input_shape else {}
    if node.op_type in {"Flatten", "Reshape", "Transpose"} and node.outputs:
        shape = next(iter(node.output_shapes.values()), None)
        return {node.outputs[0]: list(shape)} if shape else {}
    return {}


def _infer_conv_output_shape(node: NodeInfo) -> list[int | str | None] | None:
    input_name = node.inputs[0] if node.inputs else ""
    weight_index = 3 if node.op_type == "QLinearConv" else 1
    weight_name = node.inputs[weight_index] if len(node.inputs) > weight_index else ""
    input_shape = node.input_shapes.get(input_name)
    weight_shape = node.input_shapes.get(weight_name)
    if not input_shape or not weight_shape:
        return None
    if len(input_shape) not in {3, 4} or len(weight_shape) not in {3, 4}:
        return None
    if not all(isinstance(dim, int) and dim > 0 for dim in input_shape):
        return None
    if not all(isinstance(dim, int) and dim > 0 for dim in weight_shape):
        return None

    attrs = node.normalized_attributes
    strides = _as_int_list(attrs.get("strides"), [1, 1])
    pads = _as_int_list(attrs.get("pads"), [0, 0, 0, 0])
    dilations = _as_int_list(attrs.get("dilations"), [1, 1])
    output_channels = int(weight_shape[0])
    if len(input_shape) == 3:
        batch, _, input_w = [int(dim) for dim in input_shape]
        kernel_w = int(weight_shape[-1])
        stride_w = int(strides[-1] if strides else 1)
        dilation_w = int(dilations[-1] if dilations else 1)
        pad_left = int(pads[0] if pads else 0)
        pad_right = int(pads[1] if len(pads) > 1 else pad_left)
        output_w = _conv_extent(input_w, kernel_w, stride_w, dilation_w, pad_left, pad_right)
        return [batch, output_channels, output_w] if output_w is not None else None

    batch, _, input_h, input_w = [int(dim) for dim in input_shape]
    _, _, kernel_h, kernel_w = [int(dim) for dim in weight_shape]
    stride_h = int(strides[0] if strides else 1)
    stride_w = int(strides[1] if len(strides) > 1 else stride_h)
    dilation_h = int(dilations[0] if dilations else 1)
    dilation_w = int(dilations[1] if len(dilations) > 1 else dilation_h)
    pad_top = int(pads[0] if pads else 0)
    pad_left = int(pads[1] if len(pads) > 1 else pad_top)
    pad_bottom = int(pads[2] if len(pads) > 2 else pad_top)
    pad_right = int(pads[3] if len(pads) > 3 else pad_left)
    output_h = _conv_extent(input_h, kernel_h, stride_h, dilation_h, pad_top, pad_bottom)
    output_w = _conv_extent(input_w, kernel_w, stride_w, dilation_w, pad_left, pad_right)
    if output_h is None or output_w is None:
        return None
    return [batch, output_channels, output_h, output_w]


def _first_input_shape(node: NodeInfo) -> list[int | str | None] | None:
    for name in node.inputs:
        shape = node.input_shapes.get(name)
        if shape:
            return shape
    return None


def _infer_concat_output_shape(node: NodeInfo) -> list[int | str | None] | None:
    input_shapes = [node.input_shapes.get(name) for name in node.inputs]
    if not input_shapes or any(not shape for shape in input_shapes):
        return None
    rank = len(input_shapes[0] or [])
    if rank == 0 or any(len(shape or []) != rank for shape in input_shapes):
        return None
    axis = int(node.normalized_attributes.get("axis", node.attributes.get("axis", 0)))
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        return None
    output = list(input_shapes[0] or [])
    concat_dim = 0
    for shape in input_shapes:
        dim = shape[axis] if shape else None
        if not isinstance(dim, int):
            return None
        concat_dim += dim
        for index, other_dim in enumerate(shape):
            if index == axis:
                continue
            if output[index] != other_dim:
                return None
    output[axis] = concat_dim
    return output


def _conv_extent(
    input_size: int,
    kernel_size: int,
    stride: int,
    dilation: int,
    pad_before: int,
    pad_after: int,
) -> int | None:
    if stride <= 0 or dilation <= 0:
        return None
    effective_kernel = dilation * (kernel_size - 1) + 1
    output = math.floor((input_size + pad_before + pad_after - effective_kernel) / stride) + 1
    return output if output > 0 else None


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
    if op_type in {"Conv", "QLinearConv"}:
        weight_shape = weight_infos[0].shape if weight_infos else []
        spatial_rank = max(len(weight_shape) - 2, 0)
        kernel_shape = _as_int_list(attributes.get("kernel_shape"))
        if not kernel_shape and spatial_rank:
            kernel_shape = weight_shape[2:]
        spatial_rank = len(kernel_shape) or spatial_rank or 2
        return {
            "auto_pad": str(attributes.get("auto_pad", "NOTSET")),
            "kernel_shape": kernel_shape,
            "strides": _as_int_list(attributes.get("strides"), [1] * spatial_rank),
            "pads": _as_int_list(attributes.get("pads"), [0] * (spatial_rank * 2)),
            "dilations": _as_int_list(attributes.get("dilations"), [1] * spatial_rank),
            "group": int(attributes.get("group", 1)),
        }
    if op_type in {"MaxPool", "AveragePool"}:
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
    if op_type in {"Add", "QLinearAdd"}:
        return {
            "broadcast": "numpy",
        }
    if op_type == "Constant":
        return attributes
    if op_type == "Concat":
        return {"axis": int(attributes.get("axis", 0))}
    if op_type == "Gather":
        return {"axis": int(attributes.get("axis", 0))}
    if op_type == "Unsqueeze":
        return {"axes": _as_int_list(attributes.get("axes"))}
    if op_type == "Slice":
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
    if op_type in {"GlobalAveragePool", "QLinearGlobalAveragePool"}:
        return {
            "spatial_axes": "all_axes_after_batch_and_channel",
        }
    if op_type == "Softmax":
        return {
            "axis": int(attributes.get("axis", -1)),
        }
    if op_type == "Reshape":
        return {
            "allowzero": int(attributes.get("allowzero", 0)),
        }
    if op_type in {"QuantizeLinear", "DequantizeLinear"}:
        return {
            "axis": None if "axis" not in attributes else int(attributes["axis"]),
            "block_size": None
            if "block_size" not in attributes
            else int(attributes["block_size"]),
        }
    return attributes


def _as_int_list(value: Any, default: list[int] | None = None) -> list[int]:
    if value is None:
        return list(default or [])
    if isinstance(value, int):
        return [value]
    return [int(item) for item in value]


def _extract_quantization(
    graph: GraphProto,
    initializers: list[InitializerInfo],
    nodes: list[NodeInfo],
    warnings: list[str],
) -> dict[str, Any]:
    initializer_by_name = {item.name: item for item in initializers}
    qdq_infos: dict[str, dict[str, Any]] = {}
    tensor_quant: dict[str, dict[str, Any]] = {}
    dq_aliases: dict[str, str] = {}
    quantized_weights: dict[str, dict[str, Any]] = {}

    for node in graph.node:
        if node.op_type not in {"QuantizeLinear", "DequantizeLinear"}:
            continue
        info = _qdq_info(node, initializer_by_name)
        if info is None:
            warnings.append(
                f"node '{node.name or node.op_type}' is {node.op_type} but scale/zero_point "
                "is not constant; quantization extraction skipped for this boundary."
            )
            continue
        qdq_infos[node.name or f"{node.op_type}_{len(qdq_infos)}"] = info
        if node.op_type == "QuantizeLinear":
            _attach_tensor_quant(
                tensor_quant,
                node.input[0],
                info,
                source=node.name or node.op_type,
            )
            _attach_tensor_quant(
                tensor_quant,
                node.output[0],
                info,
                source=node.name or node.op_type,
            )
        else:
            quantized_input = node.input[0]
            dequantized_output = node.output[0]
            dq_aliases[dequantized_output] = quantized_input
            _attach_tensor_quant(
                tensor_quant,
                dequantized_output,
                info,
                source=node.name or node.op_type,
            )
            _attach_tensor_quant(
                tensor_quant,
                quantized_input,
                info,
                source=node.name or node.op_type,
            )
            initializer = initializer_by_name.get(quantized_input)
            if initializer is not None and initializer.elem_type in {"INT8", "UINT8"}:
                quantized_weights[quantized_input] = _quantized_weight_info(initializer, info)

    _propagate_passthrough_quant(nodes, tensor_quant)
    _propagate_fused_activation_quant(nodes, tensor_quant)
    _propagate_pool_boundary_quant(nodes, tensor_quant)

    node_quant: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if node.op_type == "QLinearConv":
            qlinear_conv_quant = _qlinear_conv_quant_info(node, initializer_by_name, warnings)
            if qlinear_conv_quant is not None:
                node_quant[node.name] = qlinear_conv_quant
                _attach_qlinear_node_tensors(tensor_quant, qlinear_conv_quant, node.name)
                _attach_qlinear_node_weight(quantized_weights, qlinear_conv_quant)
        elif node.op_type == "QLinearMatMul":
            qlinear_fc_quant = _qlinear_matmul_quant_info(node, initializer_by_name, warnings)
            if qlinear_fc_quant is not None:
                node_quant[node.name] = qlinear_fc_quant
                _attach_qlinear_node_tensors(tensor_quant, qlinear_fc_quant, node.name)
                _attach_qlinear_node_weight(quantized_weights, qlinear_fc_quant)
        elif node.op_type == "QLinearAdd":
            qlinear_add_quant = _qlinear_add_quant_info(node, initializer_by_name, warnings)
            if qlinear_add_quant is not None:
                node_quant[node.name] = qlinear_add_quant
                _attach_qlinear_node_tensors(tensor_quant, qlinear_add_quant, node.name)
                _attach_qlinear_node_weight(quantized_weights, qlinear_add_quant)
        elif node.op_type == "QLinearGlobalAveragePool":
            qlinear_pool_quant = _qlinear_global_avgpool_quant_info(
                node,
                initializer_by_name,
            )
            if qlinear_pool_quant is not None:
                node_quant[node.name] = qlinear_pool_quant
                _attach_qlinear_node_tensors(tensor_quant, qlinear_pool_quant, node.name)
        elif node.op_type in {"Gemm", "MatMul"}:
            fc_quant = _fully_connected_quant_info(
                node,
                initializer_by_name,
                tensor_quant,
                dq_aliases,
                quantized_weights,
                warnings,
            )
            if fc_quant is not None:
                node_quant[node.name] = fc_quant
        elif node.op_type == "Conv":
            conv_quant = _conv_quant_info(
                node,
                initializer_by_name,
                tensor_quant,
                dq_aliases,
                quantized_weights,
                warnings,
            )
            if conv_quant is not None:
                node_quant[node.name] = conv_quant
        elif node.op_type in {"MaxPool", "AveragePool", "GlobalAveragePool"}:
            pool_quant = _pool_quant_info(node, tensor_quant, warnings)
            if pool_quant is not None:
                node_quant[node.name] = pool_quant
        elif node.op_type == "Concat":
            concat_quant = _concat_quant_info(node, tensor_quant, warnings)
            if concat_quant is not None:
                node_quant[node.name] = concat_quant
        elif node.op_type == "Softmax":
            softmax_quant = _softmax_quant_info(node, tensor_quant)
            if softmax_quant is not None:
                node_quant[node.name] = softmax_quant

    if not tensor_quant and not node_quant and not quantized_weights:
        return {}

    return {
        "schema_version": "1.0",
        "format": "onnx_qdq",
        "granularity": "per_tensor_first_pass",
        "int8_contract": _int8_contract(nodes, node_quant),
        "tensors": tensor_quant,
        "weights": quantized_weights,
        "nodes": node_quant,
        "qdq_nodes": qdq_infos,
    }


def _int8_contract(
    nodes: list[NodeInfo],
    node_quant: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    runtime_op_types = {
        "Add",
        "AveragePool",
        "Concat",
        "Conv",
        "Gemm",
        "GlobalAveragePool",
        "MatMul",
        "MaxPool",
        "QLinearAdd",
        "QLinearConv",
        "QLinearGlobalAveragePool",
        "QLinearMatMul",
        "Softmax",
    }
    runtime_nodes = [
        node
        for node in nodes
        if node.op_type in runtime_op_types and not _is_shape_helper_concat(node)
    ]
    unsupported_nodes = [node for node in runtime_nodes if node.status == "unsupported"]
    missing_quant_nodes = [
        node
        for node in runtime_nodes
        if node.status != "unsupported" and node.name not in node_quant
    ]
    issues: list[dict[str, Any]] = []
    for node in unsupported_nodes:
        issues.append(
            {
                "node": node.name,
                "op_type": node.op_type,
                "reason": "converter marked this runtime op as unsupported",
            }
        )
    for node in missing_quant_nodes:
        issues.append(
            {
                "node": node.name,
                "op_type": node.op_type,
                "reason": (
                    "runtime op has no extracted int8 Q/DQ quantization fields for "
                    "CMSIS-NN codegen"
                ),
            }
        )
    if unsupported_nodes:
        status = "unsupported"
    elif missing_quant_nodes:
        status = "blocked"
    elif not runtime_nodes:
        status = "ok"
    elif not node_quant:
        status = "blocked"
    else:
        status = "ok"
    return {
        "schema_version": "1.0",
        "status": status,
        "format": "onnx_qdq",
        "dtype": "int8",
        "runtime_node_count": len(runtime_nodes),
        "quantized_runtime_node_count": len(node_quant),
        "issues": issues,
        "note": (
            "This contract only describes converter-side int8 Q/DQ extraction. "
            "Codegen may still block if a CMSIS-NN renderer is not implemented."
        ),
    }


def _is_shape_helper_concat(node: NodeInfo) -> bool:
    if node.op_type != "Concat":
        return False
    shapes = [
        *[shape for shape in node.input_shapes.values()],
        *[shape for shape in node.output_shapes.values()],
    ]
    return bool(shapes) and all(len(shape) <= 1 for shape in shapes)


def _qdq_info(
    node: onnx.NodeProto,
    initializer_by_name: dict[str, InitializerInfo],
) -> dict[str, Any] | None:
    if len(node.input) < 3:
        return None
    scale = initializer_by_name.get(node.input[1])
    zero_point = initializer_by_name.get(node.input[2])
    if scale is None or zero_point is None:
        return None
    scale_array = np.asarray(scale.array, dtype=np.float64).reshape(-1)
    zero_array = np.asarray(zero_point.array).reshape(-1)
    if scale_array.size != 1 or zero_array.size != 1:
        return None
    attrs = extract_attributes(node.attribute)
    zero_point_value = int(zero_array[0])
    zero_point_dtype = zero_point.elem_type
    if zero_point_dtype == "UINT8":
        zero_point_value -= 128
        zero_point_dtype = "INT8"
    return {
        "op_type": node.op_type,
        "input": node.input[0],
        "output": node.output[0] if node.output else "",
        "scale": float(scale_array[0]),
        "zero_point": zero_point_value,
        "zero_point_dtype": zero_point_dtype,
        "scale_tensor": scale.name,
        "zero_point_tensor": zero_point.name,
        "axis": None if "axis" not in attrs else int(attrs["axis"]),
    }


def _attach_tensor_quant(
    tensors: dict[str, dict[str, Any]],
    tensor_name: str,
    info: dict[str, Any],
    *,
    source: str,
) -> None:
    tensors[tensor_name] = {
        "scale": info["scale"],
        "zero_point": info["zero_point"],
        "zero_point_dtype": info["zero_point_dtype"],
        "scale_tensor": info["scale_tensor"],
        "zero_point_tensor": info["zero_point_tensor"],
        "axis": info["axis"],
        "source": source,
    }


def _propagate_passthrough_quant(
    nodes: list[NodeInfo],
    tensors: dict[str, dict[str, Any]],
) -> None:
    passthrough_ops = {"Dropout", "Flatten", "Reshape", "Transpose"}
    changed = True
    while changed:
        changed = False
        for node in nodes:
            if node.op_type not in passthrough_ops or not node.inputs or not node.outputs:
                continue
            input_quant = tensors.get(node.inputs[0])
            if input_quant is None:
                continue
            for output_name in node.outputs:
                if output_name in tensors:
                    continue
                copied = dict(input_quant)
                copied["source"] = node.name or node.op_type
                tensors[output_name] = copied
                changed = True


def _propagate_fused_activation_quant(
    nodes: list[NodeInfo],
    tensors: dict[str, dict[str, Any]],
) -> None:
    for node in nodes:
        if node.op_type not in {"Relu", "Clip"} or not node.inputs or not node.outputs:
            continue
        input_name = node.inputs[0]
        output_quant = tensors.get(node.outputs[0])
        if input_name in tensors or output_quant is None:
            continue
        copied = dict(output_quant)
        copied["source"] = node.name or node.op_type
        copied["fused_activation"] = node.op_type
        tensors[input_name] = copied


def _propagate_pool_boundary_quant(
    nodes: list[NodeInfo],
    tensors: dict[str, dict[str, Any]],
) -> None:
    changed = True
    while changed:
        changed = False
        for node in nodes:
            if node.op_type not in {"MaxPool", "AveragePool", "GlobalAveragePool"}:
                continue
            if not node.inputs or not node.outputs:
                continue
            input_name = node.inputs[0]
            output_name = node.outputs[0]
            if input_name in tensors:
                continue
            output_quant = tensors.get(output_name)
            if output_quant is None:
                continue
            copied = dict(output_quant)
            copied["source"] = node.name or node.op_type
            copied["pool_boundary_inferred"] = True
            tensors[input_name] = copied
            changed = True


def _quantized_weight_info(
    initializer: InitializerInfo,
    quant_info: dict[str, Any],
) -> dict[str, Any]:
    values = np.asarray(initializer.array).reshape(-1)
    return {
        "name": initializer.name,
        "elem_type": initializer.elem_type,
        "shape": initializer.shape,
        "element_count": initializer.element_count,
        "scale": quant_info["scale"],
        "zero_point": quant_info["zero_point"],
        "values": [int(item) for item in values.tolist()],
    }


def _attach_qlinear_node_tensors(
    tensors: dict[str, dict[str, Any]],
    node_quant: dict[str, Any],
    source: str,
) -> None:
    for section in ("inputs", "outputs"):
        values = node_quant.get(section, {})
        if not isinstance(values, dict):
            continue
        for tensor_name, quant in values.items():
            if tensor_name not in tensors and isinstance(quant, dict):
                copied = dict(quant)
                copied["source"] = source
                tensors[tensor_name] = copied


def _attach_qlinear_node_weight(
    weights: dict[str, dict[str, Any]],
    node_quant: dict[str, Any],
) -> None:
    quant_weights = node_quant.get("weights", {})
    if not isinstance(quant_weights, dict):
        return
    weight_name = str(quant_weights.get("weight", ""))
    weight_info = quant_weights.get("weight_info")
    if weight_name and isinstance(weight_info, dict):
        weights[weight_name] = weight_info


def _qlinear_conv_quant_info(
    node: NodeInfo,
    initializer_by_name: dict[str, InitializerInfo],
    warnings: list[str],
) -> dict[str, Any] | None:
    if len(node.inputs) < 8:
        return None
    x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp = node.inputs[:8]
    bias_name = node.inputs[8] if len(node.inputs) > 8 else ""
    input_quant = _qlinear_quant(initializer_by_name, x_scale, x_zp)
    weight_quant = _qlinear_quant(initializer_by_name, w_scale, w_zp, activation=False)
    output_quant = _qlinear_quant(initializer_by_name, y_scale, y_zp)
    weight_initializer = initializer_by_name.get(w)
    if (
        input_quant is None
        or weight_quant is None
        or output_quant is None
        or weight_initializer is None
    ):
        return None
    weight_values = _qlinear_int_values(weight_initializer)
    weight_shape = list(weight_initializer.shape)
    if len(weight_shape) not in {3, 4}:
        warnings.append(f"node '{node.name}' QLinearConv weight shape is not OIW/OIHW.")
        return None
    output_channels = int(weight_shape[0])
    multipliers, shifts, real_multipliers = _per_channel_requant(
        input_quant["scale"],
        weight_quant["scale"],
        output_quant["scale"],
        output_channels,
    )
    bias_values = _qlinear_bias_values(
        initializer_by_name.get(bias_name),
        expected=output_channels,
    )

    attrs = node.normalized_attributes
    raw_pads = [int(item) for item in attrs.get("pads", [0, 0, 0, 0])]
    raw_strides = [int(item) for item in attrs.get("strides", [1, 1])]
    raw_dilations = [int(item) for item in attrs.get("dilations", [1, 1])]
    raw_pads = _effective_conv_pads(node, weight_shape, raw_pads, raw_strides, raw_dilations)
    if len(weight_shape) == 3:
        pads = [
            0,
            raw_pads[0] if raw_pads else 0,
            0,
            raw_pads[1] if len(raw_pads) > 1 else 0,
        ]
        strides = [1, raw_strides[0] if raw_strides else 1]
        dilations = [1, raw_dilations[0] if raw_dilations else 1]
        weight_layout = "OIW"
        cmsis_weight_layout = "OHWI(height=1)"
    else:
        pads = raw_pads
        strides = raw_strides
        dilations = raw_dilations
        weight_layout = "OIHW"
        cmsis_weight_layout = "OHWI"

    qmin, qmax = _activation_range(output_quant)
    return {
        "op_type": node.op_type,
        "inputs": {
            x: input_quant,
            w: {
                "quantized_initializer": w,
                "scale": weight_quant["scale"],
                "zero_point": weight_quant["zero_point"],
                "zero_point_dtype": weight_quant["zero_point_dtype"],
            },
        },
        "outputs": {node.outputs[0]: output_quant} if node.outputs else {},
        "weights": {
            "weight": w,
            "bias": bias_name,
            "bias_values": bias_values,
            "weight_info": {
                "name": w,
                "elem_type": "INT8",
                "shape": weight_shape,
                "element_count": len(weight_values),
                "scale": weight_quant["scale"],
                "zero_point": weight_quant["zero_point"],
                "values": weight_values,
            },
        },
        "cmsis_nn": {
            "api": "arm_convolve_wrapper_s8",
            "real_multiplier": real_multipliers,
            "input_offset": -int(input_quant["zero_point"]),
            "output_offset": int(output_quant["zero_point"]),
            "multiplier": multipliers,
            "shift": shifts,
            "shift_semantics": (
                "CMSIS-NN arm_nn_requantize shift; positive is left shift, "
                "negative is right shift"
            ),
            "activation_min": qmin,
            "activation_max": qmax,
            "stride": strides,
            "padding": [pads[1], pads[0]] if len(pads) >= 4 else [0, 0],
            "dilation": dilations,
            "groups": int(attrs.get("group", 1)),
            "scratch_getter": "arm_convolve_wrapper_s8_get_buffer_size",
            "weight_layout": weight_layout,
            "cmsis_weight_layout": cmsis_weight_layout,
        },
    }


def _qlinear_matmul_quant_info(
    node: NodeInfo,
    initializer_by_name: dict[str, InitializerInfo],
    warnings: list[str],
) -> dict[str, Any] | None:
    if len(node.inputs) < 8:
        return None
    x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp = node.inputs[:8]
    input_quant = _qlinear_quant(initializer_by_name, x_scale, x_zp)
    weight_quant = _qlinear_quant(initializer_by_name, w_scale, w_zp, activation=False)
    output_quant = _qlinear_quant(initializer_by_name, y_scale, y_zp)
    weight_initializer = initializer_by_name.get(w)
    if (
        input_quant is None
        or weight_quant is None
        or output_quant is None
        or weight_initializer is None
    ):
        return None
    if len(weight_initializer.shape) != 2:
        warnings.append(f"node '{node.name}' QLinearMatMul weight is not rank-2.")
        return None
    input_size, output_size = [int(item) for item in weight_initializer.shape]
    raw_values = _qlinear_int_values(weight_initializer)
    weight_scales = _repeat_scales(weight_quant["scale"], output_size)
    transposed_values = []
    for out_index in range(output_size):
        for in_index in range(input_size):
            transposed_values.append(raw_values[in_index * output_size + out_index])
    multipliers, shifts, real_multipliers = _per_channel_requant(
        input_quant["scale"],
        weight_scales,
        output_quant["scale"],
        output_size,
    )
    qmin, qmax = _activation_range(output_quant)
    return {
        "op_type": node.op_type,
        "inputs": {
            x: input_quant,
            w: {
                "quantized_initializer": w,
                "scale": weight_quant["scale"],
                "zero_point": weight_quant["zero_point"],
                "zero_point_dtype": weight_quant["zero_point_dtype"],
            },
        },
        "outputs": {node.outputs[0]: output_quant} if node.outputs else {},
        "weights": {
            "weight": w,
            "bias": "",
            "bias_values": [0] * output_size,
            "weight_info": {
                "name": w,
                "elem_type": "INT8",
                "shape": [output_size, input_size],
                "element_count": len(transposed_values),
                "scale": weight_quant["scale"],
                "zero_point": 0,
                "values": transposed_values,
            },
        },
        "cmsis_nn": {
            "api": "arm_fully_connected_per_channel_s8",
            "real_multiplier": real_multipliers,
            "input_offset": -int(input_quant["zero_point"]),
            "filter_offset": 0,
            "output_offset": int(output_quant["zero_point"]),
            "multiplier": multipliers,
            "shift": shifts,
            "shift_semantics": (
                "CMSIS-NN arm_nn_requantize shift; positive is left shift, "
                "negative is right shift"
            ),
            "activation_min": qmin,
            "activation_max": qmax,
            "scratch_getter": "arm_fully_connected_s8_get_buffer_size",
        },
    }


def _qlinear_add_quant_info(
    node: NodeInfo,
    initializer_by_name: dict[str, InitializerInfo],
    warnings: list[str],
) -> dict[str, Any] | None:
    if len(node.inputs) < 8:
        return None
    a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp = node.inputs[:8]
    input_1_quant = _qlinear_quant(initializer_by_name, a_scale, a_zp)
    input_2_quant = _qlinear_quant(initializer_by_name, b_scale, b_zp)
    output_quant = _qlinear_quant(initializer_by_name, y_scale, y_zp)
    if input_1_quant is None or input_2_quant is None or output_quant is None:
        return None
    input_2_initializer = initializer_by_name.get(b)
    weight_info = None
    if input_2_initializer is not None:
        values = _qlinear_int_values(input_2_initializer)
        weight_info = {
            "name": b,
            "elem_type": "INT8",
            "shape": list(input_2_initializer.shape),
            "element_count": len(values),
            "scale": input_2_quant["scale"],
            "zero_point": input_2_quant["zero_point"],
            "values": values,
        }
    block_size = 1
    output_shape = next(iter(node.output_shapes.values()), [])
    for dim in output_shape:
        if isinstance(dim, int) and dim > 0:
            block_size *= dim
    if block_size <= 0:
        warnings.append(f"node '{node.name}' QLinearAdd output shape is dynamic.")
        return None
    left_shift = 0
    input_1_multiplier, input_1_shift = _quantize_multiplier(
        float(input_1_quant["scale"]) / float(output_quant["scale"])
    )
    input_2_multiplier, input_2_shift = _quantize_multiplier(
        float(input_2_quant["scale"]) / float(output_quant["scale"])
    )
    output_multiplier, output_shift = _quantize_multiplier(1.0)
    qmin, qmax = _activation_range(output_quant)
    weights = {"weight": "", "bias": "", "bias_values": []}
    if weight_info is not None:
        weights["weight"] = b
        weights["weight_info"] = weight_info
    return {
        "op_type": node.op_type,
        "inputs": {a: input_1_quant, b: input_2_quant},
        "outputs": {node.outputs[0]: output_quant} if node.outputs else {},
        "weights": weights,
        "cmsis_nn": {
            "api": "arm_elementwise_add_s8",
            "input_1_offset": -int(input_1_quant["zero_point"]),
            "input_1_multiplier": input_1_multiplier,
            "input_1_shift": input_1_shift,
            "input_2_offset": -int(input_2_quant["zero_point"]),
            "input_2_multiplier": input_2_multiplier,
            "input_2_shift": input_2_shift,
            "left_shift": left_shift,
            "output_offset": int(output_quant["zero_point"]),
            "output_multiplier": output_multiplier,
            "output_shift": output_shift,
            "activation_min": qmin,
            "activation_max": qmax,
            "block_size": block_size,
            "constant_input": b if weight_info is not None else "",
        },
    }


def _qlinear_global_avgpool_quant_info(
    node: NodeInfo,
    initializer_by_name: dict[str, InitializerInfo],
) -> dict[str, Any] | None:
    if len(node.inputs) < 5:
        return None
    x, x_scale, x_zp, y_scale, y_zp = node.inputs[:5]
    input_quant = _qlinear_quant(initializer_by_name, x_scale, x_zp)
    output_quant = _qlinear_quant(initializer_by_name, y_scale, y_zp)
    if input_quant is None or output_quant is None:
        return None
    input_shape = node.input_shapes.get(x, [])
    if len(input_shape) != 4:
        return None
    quantization_differs = (
        float(input_quant["scale"]) != float(output_quant["scale"])
        or int(input_quant["zero_point"]) != int(output_quant["zero_point"])
    )
    requant_multiplier = 0
    requant_shift = 0
    if quantization_differs:
        requant_multiplier, requant_shift = _quantize_multiplier(
            float(input_quant["scale"]) / float(output_quant["scale"])
        )
    qmin, qmax = _activation_range(output_quant)
    return {
        "op_type": node.op_type,
        "inputs": {x: input_quant},
        "outputs": {node.outputs[0]: output_quant} if node.outputs else {},
        "cmsis_nn": {
            "api": "arm_avgpool_s8",
            "stride": [int(input_shape[2]), int(input_shape[3])],
            "padding": [0, 0],
            "kernel_shape": [int(input_shape[2]), int(input_shape[3])],
            "activation_min": qmin,
            "activation_max": qmax,
            "scratch_getter": "arm_avgpool_s8_get_buffer_size",
            "requantize_output": quantization_differs,
            "requantize_multiplier": requant_multiplier,
            "requantize_shift": requant_shift,
            "requantize_input_zero_point": int(input_quant["zero_point"]),
            "requantize_output_zero_point": int(output_quant["zero_point"]),
        },
    }


def _qlinear_quant(
    initializer_by_name: dict[str, InitializerInfo],
    scale_name: str,
    zero_point_name: str,
    *,
    activation: bool = True,
) -> dict[str, Any] | None:
    scale = initializer_by_name.get(scale_name)
    zero_point = initializer_by_name.get(zero_point_name)
    if scale is None or zero_point is None:
        return None
    scale_array = np.asarray(scale.array, dtype=np.float64).reshape(-1)
    zero_array = np.asarray(zero_point.array).reshape(-1)
    if scale_array.size == 0 or zero_array.size == 0:
        return None
    scale_value: float | list[float]
    zero_value: int | list[int]
    if scale_array.size == 1:
        scale_value = float(scale_array[0])
    else:
        scale_value = [float(item) for item in scale_array.tolist()]
    if zero_array.size == 1:
        zero_value = int(zero_array[0])
    else:
        zero_value = [int(item) for item in zero_array.tolist()]
    dtype = zero_point.elem_type
    if activation and dtype == "UINT8":
        zero_value = _shift_uint8_zero_point(zero_value)
        dtype = "INT8"
    return {
        "scale": scale_value,
        "zero_point": zero_value,
        "zero_point_dtype": dtype,
        "scale_tensor": scale.name,
        "zero_point_tensor": zero_point.name,
        "axis": 0 if scale_array.size > 1 else None,
        "source": "qlinear",
    }


def _shift_uint8_zero_point(value: int | list[int]) -> int | list[int]:
    if isinstance(value, list):
        return [int(item) - 128 for item in value]
    return int(value) - 128


def _qlinear_int_values(initializer: InitializerInfo) -> list[int]:
    values = np.asarray(initializer.array).reshape(-1)
    if initializer.elem_type == "UINT8":
        return [int(item) - 128 for item in values.tolist()]
    return [int(item) for item in values.tolist()]


def _qlinear_bias_values(
    bias: InitializerInfo | None,
    *,
    expected: int,
) -> list[int]:
    if bias is None:
        return [0] * expected
    values = [int(item) for item in np.asarray(bias.array).reshape(-1).tolist()]
    if len(values) == expected:
        return values
    return (values + [0] * expected)[:expected]


def _effective_conv_pads(
    node: NodeInfo,
    weight_shape: list[int],
    raw_pads: list[int],
    strides: list[int],
    dilations: list[int],
) -> list[int]:
    """Return explicit ONNX pads, deriving SAME_* auto_pad when needed."""
    auto_pad = str(node.normalized_attributes.get("auto_pad", "NOTSET")).upper()
    if auto_pad in {"", "NOTSET"}:
        return raw_pads
    if auto_pad == "VALID":
        return [0] * (2 if len(weight_shape) == 3 else 4)
    if auto_pad not in {"SAME_UPPER", "SAME_LOWER"}:
        return raw_pads

    input_name = node.inputs[0] if node.inputs else ""
    output_name = node.outputs[0] if node.outputs else ""
    input_shape = node.input_shapes.get(input_name, [])
    output_shape = node.output_shapes.get(output_name, [])
    if len(weight_shape) == 3:
        if len(input_shape) != 3 or len(output_shape) != 3:
            return raw_pads
        input_sizes = [int(input_shape[2])]
        output_sizes = [int(output_shape[2])]
        kernel_sizes = [int(weight_shape[2])]
    elif len(weight_shape) == 4:
        if len(input_shape) != 4 or len(output_shape) != 4:
            return raw_pads
        input_sizes = [int(input_shape[2]), int(input_shape[3])]
        output_sizes = [int(output_shape[2]), int(output_shape[3])]
        kernel_sizes = [int(weight_shape[2]), int(weight_shape[3])]
    else:
        return raw_pads

    before: list[int] = []
    after: list[int] = []
    for index, (input_size, output_size, kernel_size) in enumerate(
        zip(input_sizes, output_sizes, kernel_sizes, strict=True)
    ):
        stride = int(strides[index] if index < len(strides) else strides[-1] if strides else 1)
        dilation = int(
            dilations[index] if index < len(dilations) else dilations[-1] if dilations else 1
        )
        effective_kernel = dilation * (kernel_size - 1) + 1
        total_pad = max((output_size - 1) * stride + effective_kernel - input_size, 0)
        if auto_pad == "SAME_UPPER":
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
        else:
            pad_after = total_pad // 2
            pad_before = total_pad - pad_after
        before.append(pad_before)
        after.append(pad_after)

    return [*before, *after]


def _per_channel_requant(
    input_scale: float | list[float],
    weight_scale: float | list[float],
    output_scale: float | list[float],
    channels: int,
) -> tuple[list[int], list[int], list[float]]:
    input_scales = _repeat_scales(input_scale, channels)
    weight_scales = _repeat_scales(weight_scale, channels)
    output_scales = _repeat_scales(output_scale, channels)
    multipliers: list[int] = []
    shifts: list[int] = []
    real_multipliers: list[float] = []
    for in_scale, w_scale, out_scale in zip(
        input_scales,
        weight_scales,
        output_scales,
        strict=True,
    ):
        real = float(in_scale) * float(w_scale) / float(out_scale)
        multiplier, shift = _quantize_multiplier(real)
        real_multipliers.append(real)
        multipliers.append(multiplier)
        shifts.append(shift)
    return multipliers, shifts, real_multipliers


def _repeat_scales(value: float | list[float], count: int) -> list[float]:
    if isinstance(value, list):
        if len(value) == count:
            return [float(item) for item in value]
        if len(value) == 1:
            return [float(value[0])] * count
        return [float(value[min(index, len(value) - 1)]) for index in range(count)]
    return [float(value)] * count


def _fully_connected_quant_info(
    node: NodeInfo,
    initializer_by_name: dict[str, InitializerInfo],
    tensor_quant: dict[str, dict[str, Any]],
    dq_aliases: dict[str, str],
    quantized_weights: dict[str, dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any] | None:
    if len(node.inputs) < 2:
        return None
    if node.op_type == "Gemm" and int(node.normalized_attributes.get("transB", 0)) != 1:
        warnings.append(
            f"node '{node.name}' is quantized Gemm but transB is not 1; first CMSIS-NN "
            "FC renderer requires row-major [out, in] weights."
        )
        return None
    input_name = node.inputs[0]
    weight_input = node.inputs[1]
    output_name = node.outputs[0] if node.outputs else ""
    quant_weight_name = dq_aliases.get(weight_input, weight_input)
    weight_info = quantized_weights.get(quant_weight_name)
    input_quant = tensor_quant.get(input_name)
    output_quant = tensor_quant.get(output_name)
    if input_quant is None or output_quant is None or weight_info is None:
        return None
    if node.op_type == "MatMul":
        matmul_weight = _matmul_fc_weight_info(
            node,
            quant_weight_name,
            weight_info,
            warnings,
        )
        if matmul_weight is None:
            return None
        quant_weight_name, weight_info = matmul_weight
        quantized_weights[quant_weight_name] = weight_info

    real_multiplier = (
        float(input_quant["scale"]) * float(weight_info["scale"]) / float(output_quant["scale"])
    )
    multiplier, shift = _quantize_multiplier(real_multiplier)
    bias_values: list[int] = []
    bias_name = node.inputs[2] if len(node.inputs) > 2 else ""
    bias = initializer_by_name.get(bias_name)
    if bias is not None:
        bias_scale = float(input_quant["scale"]) * float(weight_info["scale"])
        bias_values = _quantize_bias(bias, bias_scale, warnings)

    qmin, qmax = _activation_range(output_quant)
    return {
        "op_type": node.op_type,
        "inputs": {
            input_name: input_quant,
            weight_input: {
                "quantized_initializer": quant_weight_name,
                "scale": weight_info["scale"],
                "zero_point": weight_info["zero_point"],
                "zero_point_dtype": weight_info["elem_type"],
            },
        },
        "outputs": {output_name: output_quant},
        "weights": {
            "weight": quant_weight_name,
            "bias": bias_name,
            "bias_values": bias_values,
        },
        "cmsis_nn": {
            "api": "arm_fully_connected_s8",
            "real_multiplier": real_multiplier,
            "input_offset": -int(input_quant["zero_point"]),
            "filter_offset": 0,
            "output_offset": int(output_quant["zero_point"]),
            "multiplier": multiplier,
            "shift": shift,
            "shift_semantics": (
                "CMSIS-NN arm_nn_requantize shift; positive is left shift, "
                "negative is right shift"
            ),
            "activation_min": qmin,
            "activation_max": qmax,
            "scratch_getter": "arm_fully_connected_s8_get_buffer_size",
        },
    }


def _matmul_fc_weight_info(
    node: NodeInfo,
    quant_weight_name: str,
    weight_info: dict[str, Any],
    warnings: list[str],
) -> tuple[str, dict[str, Any]] | None:
    weight_shape = weight_info.get("shape", [])
    if not isinstance(weight_shape, list) or len(weight_shape) != 2:
        warnings.append(
            f"node '{node.name}' is quantized MatMul but constant weight is not rank-2."
        )
        return None
    input_name = node.inputs[0] if node.inputs else ""
    output_name = node.outputs[0] if node.outputs else ""
    input_shape = node.input_shapes.get(input_name, [])
    output_shape = node.output_shapes.get(output_name, [])
    if (
        len(input_shape) != 2
        or len(output_shape) != 2
        or not all(isinstance(dim, int) and dim > 0 for dim in input_shape + output_shape)
        or int(input_shape[0]) != 1
        or int(output_shape[0]) != 1
    ):
        warnings.append(
            f"node '{node.name}' is quantized MatMul but is not the supported [1,I] x [I,O] FC form."
        )
        return None
    input_size = int(input_shape[1])
    output_size = int(output_shape[1])
    if int(weight_shape[0]) != input_size or int(weight_shape[1]) != output_size:
        warnings.append(
            f"node '{node.name}' MatMul weight shape {weight_shape} does not match [I,O]=[{input_size},{output_size}]."
        )
        return None
    values = weight_info.get("values", [])
    if not isinstance(values, list) or len(values) != input_size * output_size:
        warnings.append(f"node '{node.name}' MatMul weight values are incomplete.")
        return None
    transposed_values = (
        np.asarray(values, dtype=np.int8)
        .reshape(input_size, output_size)
        .T
        .reshape(-1)
        .astype(np.int8)
        .tolist()
    )
    cmsis_weight_name = f"{quant_weight_name}__cmsis_fc"
    cmsis_weight = dict(weight_info)
    cmsis_weight["name"] = cmsis_weight_name
    cmsis_weight["shape"] = [output_size, input_size]
    cmsis_weight["values"] = [int(item) for item in transposed_values]
    cmsis_weight["source_initializer"] = quant_weight_name
    cmsis_weight["layout_transform"] = "MatMul B[I,O] -> CMSIS FC weight[O,I]"
    return cmsis_weight_name, cmsis_weight


def _conv_quant_info(
    node: NodeInfo,
    initializer_by_name: dict[str, InitializerInfo],
    tensor_quant: dict[str, dict[str, Any]],
    dq_aliases: dict[str, str],
    quantized_weights: dict[str, dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any] | None:
    if len(node.inputs) < 2:
        return None
    input_name = node.inputs[0]
    weight_input = node.inputs[1]
    output_name = node.outputs[0] if node.outputs else ""
    quant_weight_name = dq_aliases.get(weight_input, weight_input)
    weight_info = quantized_weights.get(quant_weight_name)
    input_quant = tensor_quant.get(input_name)
    output_quant = tensor_quant.get(output_name)
    if input_quant is None or output_quant is None or weight_info is None:
        return None
    weight_shape = weight_info.get("shape", [])
    if not isinstance(weight_shape, list) or len(weight_shape) not in {3, 4}:
        warnings.append(
            f"node '{node.name}' is quantized Conv but weight shape is not OIW/OIHW."
        )
        return None
    output_channels = int(weight_shape[0])
    real_multiplier = (
        float(input_quant["scale"]) * float(weight_info["scale"]) / float(output_quant["scale"])
    )
    multiplier, shift = _quantize_multiplier(real_multiplier)
    bias_values: list[int] = []
    bias_name = node.inputs[2] if len(node.inputs) > 2 else ""
    bias = initializer_by_name.get(bias_name)
    if bias is not None:
        bias_scale = float(input_quant["scale"]) * float(weight_info["scale"])
        bias_values = _quantize_bias(bias, bias_scale, warnings)
    if not bias_values:
        bias_values = [0] * output_channels

    attrs = node.normalized_attributes
    raw_pads = [int(item) for item in attrs.get("pads", [0, 0, 0, 0])]
    raw_strides = [int(item) for item in attrs.get("strides", [1, 1])]
    raw_dilations = [int(item) for item in attrs.get("dilations", [1, 1])]
    raw_pads = _effective_conv_pads(node, weight_shape, raw_pads, raw_strides, raw_dilations)
    if len(weight_shape) == 3:
        pads = [0, raw_pads[0] if raw_pads else 0, 0, raw_pads[1] if len(raw_pads) > 1 else 0]
        strides = [1, raw_strides[0] if raw_strides else 1]
        dilations = [1, raw_dilations[0] if raw_dilations else 1]
        weight_layout = "OIW"
        cmsis_weight_layout = "OHWI(height=1)"
    else:
        pads = raw_pads
        strides = raw_strides
        dilations = raw_dilations
        weight_layout = "OIHW"
        cmsis_weight_layout = "OHWI"
    qmin, qmax = _activation_range(output_quant)
    return {
        "op_type": node.op_type,
        "inputs": {
            input_name: input_quant,
            weight_input: {
                "quantized_initializer": quant_weight_name,
                "scale": weight_info["scale"],
                "zero_point": weight_info["zero_point"],
                "zero_point_dtype": weight_info["elem_type"],
            },
        },
        "outputs": {output_name: output_quant},
        "weights": {
            "weight": quant_weight_name,
            "bias": bias_name,
            "bias_values": bias_values,
        },
        "cmsis_nn": {
            "api": "arm_convolve_wrapper_s8",
            "real_multiplier": real_multiplier,
            "input_offset": -int(input_quant["zero_point"]),
            "output_offset": int(output_quant["zero_point"]),
            "multiplier": [multiplier] * output_channels,
            "shift": [shift] * output_channels,
            "shift_semantics": (
                "CMSIS-NN arm_nn_requantize shift; positive is left shift, "
                "negative is right shift"
            ),
            "activation_min": qmin,
            "activation_max": qmax,
            "stride": strides,
            "padding": [pads[1], pads[0]] if len(pads) >= 4 else [0, 0],
            "dilation": dilations,
            "groups": int(attrs.get("group", 1)),
            "scratch_getter": "arm_convolve_wrapper_s8_get_buffer_size",
            "weight_layout": weight_layout,
            "cmsis_weight_layout": cmsis_weight_layout,
        },
    }


def _pool_quant_info(
    node: NodeInfo,
    tensor_quant: dict[str, dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any] | None:
    if not node.inputs or not node.outputs:
        return None
    input_name = node.inputs[0]
    output_name = node.outputs[0]
    input_quant = tensor_quant.get(input_name)
    output_quant = tensor_quant.get(output_name)
    if input_quant is None:
        return None
    if output_quant is None:
        output_quant = dict(input_quant)
        output_quant["source"] = node.name or node.op_type
    quantization_differs = (
        float(input_quant["scale"]) != float(output_quant["scale"])
        or int(input_quant["zero_point"]) != int(output_quant["zero_point"])
    )
    if quantization_differs and node.op_type not in {"AveragePool", "GlobalAveragePool"}:
        warnings.append(
            f"node '{node.name}' is {node.op_type} but input/output quantization differs; "
            "CMSIS-NN s8 pooling does not requantize in this renderer."
        )
        return None
    requant_multiplier = 0
    requant_shift = 0
    if quantization_differs:
        requant_multiplier, requant_shift = _quantize_multiplier(
            float(input_quant["scale"]) / float(output_quant["scale"])
        )
    attrs = node.normalized_attributes
    input_shape = next(iter(node.input_shapes.values()), [])
    kernel_shape = attrs.get("kernel_shape", [])
    if node.op_type == "GlobalAveragePool":
        if len(input_shape) == 4:
            kernel_shape = [int(input_shape[2]), int(input_shape[3])]
        else:
            return None
    if not isinstance(kernel_shape, list) or len(kernel_shape) != 2:
        return None
    strides = [int(item) for item in attrs.get("strides", kernel_shape)]
    pads = [int(item) for item in attrs.get("pads", [0, 0, 0, 0])]
    qmin, qmax = _quantized_range(str(output_quant["zero_point_dtype"]))
    api = (
        "arm_avgpool_s8"
        if node.op_type in {"AveragePool", "GlobalAveragePool"}
        else "arm_max_pool_s8"
    )
    return {
        "op_type": node.op_type,
        "inputs": {input_name: input_quant},
        "outputs": {output_name: output_quant},
        "cmsis_nn": {
            "api": api,
            "stride": strides,
            "padding": [pads[1], pads[0]] if len(pads) >= 4 else [0, 0],
            "kernel_shape": [int(kernel_shape[0]), int(kernel_shape[1])],
            "activation_min": qmin,
            "activation_max": qmax,
            "scratch_getter": "arm_avgpool_s8_get_buffer_size"
            if api == "arm_avgpool_s8"
            else None,
            "requantize_output": quantization_differs,
            "requantize_multiplier": requant_multiplier,
            "requantize_shift": requant_shift,
            "requantize_input_zero_point": int(input_quant["zero_point"]),
            "requantize_output_zero_point": int(output_quant["zero_point"]),
        },
    }


def _concat_quant_info(
    node: NodeInfo,
    tensor_quant: dict[str, dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any] | None:
    if not node.inputs or not node.outputs:
        return None
    output_name = node.outputs[0]
    input_quants = [tensor_quant.get(input_name) for input_name in node.inputs]
    output_quant = tensor_quant.get(output_name)
    if output_quant is None or any(item is None for item in input_quants):
        return None

    reference = input_quants[0]
    assert reference is not None
    all_quants = [*input_quants, output_quant]
    input_requantize: list[dict[str, Any]] = []
    for quant in all_quants[1:]:
        assert quant is not None
        if (
            float(quant["scale"]) != float(reference["scale"])
            or int(quant["zero_point"]) != int(reference["zero_point"])
            or str(quant["zero_point_dtype"]) != str(reference["zero_point_dtype"])
        ):
            break
    for quant in input_quants:
        assert quant is not None
        if str(quant["zero_point_dtype"]) != str(output_quant["zero_point_dtype"]):
            warnings.append(
                f"node '{node.name}' is Concat but input/output zero_point dtype differs; "
                "this renderer only supports a single int8 dtype across concat branches."
            )
            return None
        differs = (
            float(quant["scale"]) != float(output_quant["scale"])
            or int(quant["zero_point"]) != int(output_quant["zero_point"])
        )
        multiplier = 0
        shift = 0
        if differs:
            multiplier, shift = _quantize_multiplier(
                float(quant["scale"]) / float(output_quant["scale"])
            )
        input_requantize.append(
            {
                "required": differs,
                "input_scale": quant["scale"],
                "input_zero_point": quant["zero_point"],
                "output_scale": output_quant["scale"],
                "output_zero_point": output_quant["zero_point"],
                "multiplier": multiplier,
                "shift": shift,
            }
        )

    return {
        "op_type": node.op_type,
        "inputs": {
            input_name: quant
            for input_name, quant in zip(node.inputs, input_quants, strict=True)
            if quant is not None
        },
        "outputs": {output_name: output_quant},
        "cmsis_nn": {
            "api": "arm_concatenation_s8",
            "same_quantization": not any(item["required"] for item in input_requantize),
            "scale": output_quant["scale"],
            "zero_point": output_quant["zero_point"],
            "zero_point_dtype": output_quant["zero_point_dtype"],
            "input_requantize": input_requantize,
        },
    }


def _softmax_quant_info(
    node: NodeInfo,
    tensor_quant: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if not node.inputs or not node.outputs:
        return None
    input_name = node.inputs[0]
    output_name = node.outputs[0]
    input_quant = tensor_quant.get(input_name)
    output_quant = tensor_quant.get(output_name)
    if input_quant is None:
        return None
    if output_quant is None:
        output_quant = {
            "scale": 1.0 / 256.0,
            "zero_point": -128,
            "zero_point_dtype": "INT8",
            "scale_tensor": "",
            "zero_point_tensor": "",
            "axis": None,
            "source": node.name or node.op_type,
            "softmax_output_inferred": True,
        }
        tensor_quant[output_name] = output_quant
    input_integer_bits = 5
    real_multiplier = float(input_quant["scale"]) * float(1 << (31 - input_integer_bits))
    multiplier, shift = _quantize_multiplier(real_multiplier)
    diff_min = -_input_radius(input_integer_bits, shift)
    return {
        "op_type": node.op_type,
        "inputs": {input_name: input_quant},
        "outputs": {output_name: output_quant},
        "cmsis_nn": {
            "api": "arm_softmax_s8",
            "input_scale": input_quant["scale"],
            "output_scale": output_quant["scale"],
            "output_zero_point": output_quant["zero_point"],
            "multiplier": multiplier,
            "shift": shift,
            "diff_min": diff_min,
            "input_integer_bits": input_integer_bits,
        },
    }


def _quantize_bias(
    bias: InitializerInfo,
    bias_scale: float,
    warnings: list[str],
) -> list[int]:
    if bias_scale == 0.0:
        warnings.append(f"bias '{bias.name}' cannot be quantized because bias_scale is zero.")
        return []
    values = np.asarray(bias.array, dtype=np.float64).reshape(-1)
    quantized = np.rint(values / bias_scale).astype(np.int64)
    if np.any(quantized < np.iinfo(np.int32).min) or np.any(quantized > np.iinfo(np.int32).max):
        warnings.append(f"bias '{bias.name}' quantized values exceed int32 range.")
    clipped = np.clip(quantized, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
    return [int(item) for item in clipped.tolist()]


def _quantize_multiplier(real_multiplier: float) -> tuple[int, int]:
    if real_multiplier <= 0.0 or not math.isfinite(real_multiplier):
        return 0, 0
    significand, exponent = math.frexp(real_multiplier)
    quantized = int(round(significand * (1 << 31)))
    if quantized == 1 << 31:
        quantized //= 2
        exponent += 1
    return quantized, exponent


def _quantized_range(dtype: str) -> tuple[int, int]:
    if dtype == "UINT8":
        return 0, 255
    return -128, 127


def _activation_range(output_quant: dict[str, Any]) -> tuple[int, int]:
    qmin, qmax = _quantized_range(str(output_quant["zero_point_dtype"]))
    if output_quant.get("fused_activation") == "Relu":
        qmin = max(qmin, int(output_quant["zero_point"]))
    return qmin, qmax


def _input_radius(input_integer_bits: int, input_left_shift: int) -> int:
    total_signed_bits = 31
    max_input_rescaled = ((1 << input_integer_bits) - 1) * (
        1 << (total_signed_bits - input_integer_bits)
    )
    return int(math.floor(max_input_rescaled / (1 << max(input_left_shift, 0))))
