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
    "GlobalAveragePool",
    "MatMul",
    "MaxPool",
    "DequantizeLinear",
    "QuantizeLinear",
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
    "DequantizeLinear": {0},
    "Gemm": {1, 2},
    "MatMul": {1},
    "QuantizeLinear": {0},
}
AUXILIARY_INITIALIZER_INPUTS = {
    "DequantizeLinear": {1, 2},
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

    node_quant: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if node.op_type in {"Gemm", "MatMul"}:
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
        "Conv",
        "Gemm",
        "GlobalAveragePool",
        "MatMul",
        "MaxPool",
        "Softmax",
    }
    runtime_nodes = [
        node
        for node in nodes
        if node.op_type in runtime_op_types
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
    elif missing_quant_nodes or not node_quant:
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
    return {
        "op_type": node.op_type,
        "input": node.input[0],
        "output": node.output[0] if node.output else "",
        "scale": float(scale_array[0]),
        "zero_point": int(zero_array[0]),
        "zero_point_dtype": zero_point.elem_type,
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
    passthrough_ops = {"Flatten", "Reshape", "Transpose"}
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
    if node.op_type == "MatMul":
        warnings.append(
            f"node '{node.name}' is quantized MatMul; first CMSIS-NN FC renderer only "
            "supports Gemm with transB=1."
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

    qmin, qmax = _quantized_range(str(output_quant["zero_point_dtype"]))
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
    if not isinstance(weight_shape, list) or len(weight_shape) != 4:
        warnings.append(
            f"node '{node.name}' is quantized Conv but weight shape is not OIHW rank-4."
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
    pads = [int(item) for item in attrs.get("pads", [0, 0, 0, 0])]
    strides = [int(item) for item in attrs.get("strides", [1, 1])]
    dilations = [int(item) for item in attrs.get("dilations", [1, 1])]
    qmin, qmax = _quantized_range(str(output_quant["zero_point_dtype"]))
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
            "weight_layout": "OIHW",
            "cmsis_weight_layout": "OHWI",
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
    if input_quant is None or output_quant is None:
        return None
    if (
        float(input_quant["scale"]) != float(output_quant["scale"])
        or int(input_quant["zero_point"]) != int(output_quant["zero_point"])
    ):
        warnings.append(
            f"node '{node.name}' is {node.op_type} but input/output quantization differs; "
            "CMSIS-NN s8 pooling does not requantize in this renderer."
        )
        return None
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
    if input_quant is None or output_quant is None:
        return None
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


def _input_radius(input_integer_bits: int, input_left_shift: int) -> int:
    total_signed_bits = 31
    max_input_rescaled = ((1 << input_integer_bits) - 1) * (
        1 << (total_signed_bits - input_integer_bits)
    )
    return int(math.floor(max_input_rescaled / (1 << max(input_left_shift, 0))))
