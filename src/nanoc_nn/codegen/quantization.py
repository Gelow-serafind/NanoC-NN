from __future__ import annotations

from .model import ModelGraph, OpMapping, QuantizationIssue


def analyze_quantization(
    graph: ModelGraph,
    mappings: list[OpMapping],
) -> list[QuantizationIssue]:
    issues: list[QuantizationIssue] = []
    if not graph.has_quantization:
        for mapping in mappings:
            if mapping.needs_quantization:
                issues.append(
                    QuantizationIssue(
                        node_name=mapping.node_name,
                        onnx_op=mapping.onnx_op,
                        requirement=(
                            "missing scale/zero_point/multiplier/shift information required by "
                            f"{mapping.cmsis_action}"
                        ),
                    )
                )
        return issues

    quantization = graph.quantization or {}
    tensor_quant = quantization.get("tensors", {})
    node_quant = quantization.get("nodes", {})
    for mapping in mappings:
        if not mapping.needs_quantization:
            continue
        missing = _missing_node_quant_fields(mapping, node_quant, tensor_quant)
        for field in missing:
            issues.append(
                QuantizationIssue(
                    node_name=mapping.node_name,
                    onnx_op=mapping.onnx_op,
                    requirement=f"missing quantization field: {field}",
                )
            )
    return issues


def _missing_node_quant_fields(
    mapping: OpMapping,
    node_quant: object,
    tensor_quant: object,
) -> list[str]:
    missing: list[str] = []
    if not isinstance(node_quant, dict) or mapping.node_name not in node_quant:
        missing.append(f"nodes.{mapping.node_name}")
        return missing
    node_info = node_quant[mapping.node_name]
    if not isinstance(node_info, dict):
        missing.append(f"nodes.{mapping.node_name}")
        return missing
    cmsis_nn = node_info.get("cmsis_nn")
    if not isinstance(cmsis_nn, dict):
        missing.append(f"nodes.{mapping.node_name}.cmsis_nn")
        return missing
    for field in (
        "input_offset",
        "filter_offset",
        "output_offset",
        "multiplier",
        "shift",
        "activation_min",
        "activation_max",
    ):
        if field not in cmsis_nn:
            missing.append(f"nodes.{mapping.node_name}.cmsis_nn.{field}")
    if not isinstance(tensor_quant, dict):
        missing.append("tensors")
        return missing
    inputs = node_info.get("inputs", {})
    outputs = node_info.get("outputs", {})
    if not isinstance(inputs, dict):
        missing.append(f"nodes.{mapping.node_name}.inputs")
    if not isinstance(outputs, dict):
        missing.append(f"nodes.{mapping.node_name}.outputs")
    return missing
