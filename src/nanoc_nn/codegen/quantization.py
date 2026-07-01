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
    if not isinstance(tensor_quant, dict):
        missing.append("tensors")
        return missing
    for tensor_name in [*mapping.inputs, *mapping.outputs]:
        if tensor_name and tensor_name not in tensor_quant:
            missing.append(f"tensors.{tensor_name}")
    return missing
