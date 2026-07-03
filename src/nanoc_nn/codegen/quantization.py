from __future__ import annotations

from .model import ModelGraph, OpMapping, QuantizationIssue

REQUIRED_NODE_QUANT_FIELDS = {
    "Conv": {
        "cmsis_nn": {
            "api",
            "input_offset",
            "output_offset",
            "multiplier",
            "shift",
            "activation_min",
            "activation_max",
            "stride",
            "padding",
            "dilation",
            "groups",
            "scratch_getter",
            "weight_layout",
            "cmsis_weight_layout",
        },
        "weights": {"weight", "bias", "bias_values"},
        "inputs": "non_empty_dict",
        "outputs": "non_empty_dict",
    },
    "Gemm": {
        "cmsis_nn": {
            "api",
            "input_offset",
            "filter_offset",
            "output_offset",
            "multiplier",
            "shift",
            "activation_min",
            "activation_max",
            "scratch_getter",
        },
        "weights": {"weight", "bias", "bias_values"},
        "inputs": "non_empty_dict",
        "outputs": "non_empty_dict",
    },
    "MatMul": {
        "cmsis_nn": {
            "api",
            "input_offset",
            "filter_offset",
            "output_offset",
            "multiplier",
            "shift",
            "activation_min",
            "activation_max",
            "scratch_getter",
        },
        "weights": {"weight", "bias", "bias_values"},
        "inputs": "non_empty_dict",
        "outputs": "non_empty_dict",
    },
    "MaxPool": {
        "cmsis_nn": {
            "api",
            "stride",
            "padding",
            "kernel_shape",
            "activation_min",
            "activation_max",
        },
        "inputs": "non_empty_dict",
        "outputs": "non_empty_dict",
    },
    "AveragePool": {
        "cmsis_nn": {
            "api",
            "stride",
            "padding",
            "kernel_shape",
            "activation_min",
            "activation_max",
        },
        "inputs": "non_empty_dict",
        "outputs": "non_empty_dict",
    },
    "GlobalAveragePool": {
        "cmsis_nn": {
            "api",
            "stride",
            "padding",
            "kernel_shape",
            "activation_min",
            "activation_max",
        },
        "inputs": "non_empty_dict",
        "outputs": "non_empty_dict",
    },
    "Softmax": {
        "cmsis_nn": {
            "api",
            "multiplier",
            "shift",
            "diff_min",
        },
        "inputs": "non_empty_dict",
        "outputs": "non_empty_dict",
    },
}


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
    contract = quantization.get("int8_contract")
    if isinstance(contract, dict) and contract.get("status") not in {None, "ok"}:
        issues.append(
            QuantizationIssue(
                node_name="__model__",
                onnx_op="model",
                requirement=(
                    "converter int8 contract is "
                    f"{contract.get('status')}: {_contract_issue_summary(contract)}"
                ),
            )
        )
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
    rules = REQUIRED_NODE_QUANT_FIELDS.get(mapping.onnx_op, {})
    cmsis_fields = rules.get("cmsis_nn", set())
    if not isinstance(cmsis_fields, set):
        cmsis_fields = set()
    for field in sorted(cmsis_fields):
        if field not in cmsis_nn:
            missing.append(f"nodes.{mapping.node_name}.cmsis_nn.{field}")
    if not isinstance(tensor_quant, dict):
        missing.append("tensors")
        return missing
    inputs = node_info.get("inputs", {})
    outputs = node_info.get("outputs", {})
    if rules.get("inputs") == "non_empty_dict" and (
        not isinstance(inputs, dict) or not inputs
    ):
        missing.append(f"nodes.{mapping.node_name}.inputs")
    if rules.get("outputs") == "non_empty_dict" and (
        not isinstance(outputs, dict) or not outputs
    ):
        missing.append(f"nodes.{mapping.node_name}.outputs")
    weights = node_info.get("weights", {})
    weight_fields = rules.get("weights", set())
    if isinstance(weight_fields, set):
        if not isinstance(weights, dict):
            missing.append(f"nodes.{mapping.node_name}.weights")
        else:
            for field in sorted(weight_fields):
                if field not in weights:
                    missing.append(f"nodes.{mapping.node_name}.weights.{field}")
    return missing


def _contract_issue_summary(contract: dict) -> str:
    issues = contract.get("issues")
    if not isinstance(issues, list) or not issues:
        return "no detail"
    first = issues[0]
    if not isinstance(first, dict):
        return "invalid issue detail"
    node = first.get("node", "unknown")
    reason = first.get("reason", "unknown reason")
    extra = len(issues) - 1
    suffix = f"; plus {extra} more" if extra > 0 else ""
    return f"{node}: {reason}{suffix}"
