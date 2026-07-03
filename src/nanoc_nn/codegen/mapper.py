from __future__ import annotations

from .model import ModelGraph, NodeSpec, OpMapping, element_count_from_shape

RUNTIME_ACTIONS = {
    "Add": ("direct_api", "arm_elementwise_add_s8", True),
    "AveragePool": ("direct_api", "arm_avgpool_s8", True),
    "Concat": ("direct_api", "arm_concatenation_s8_x/y/z/w", True),
    "Conv": ("wrapper_api", "arm_convolve_wrapper_s8", True),
    "Gemm": ("wrapper_api", "arm_fully_connected_wrapper_s8", True),
    "GlobalAveragePool": ("direct_api", "arm_avgpool_s8", True),
    "MatMul": ("wrapper_api", "arm_fully_connected_wrapper_s8", True),
    "MaxPool": ("direct_api", "arm_max_pool_s8", True),
    "Mul": ("direct_api", "arm_elementwise_mul_s8", True),
    "Softmax": ("direct_api", "arm_softmax_s8", True),
    "Transpose": ("direct_api", "arm_transpose_s8", True),
}

RENDERED_RUNTIME_OPS = {
    "AveragePool",
    "Conv",
    "Gemm",
    "GlobalAveragePool",
    "MatMul",
    "MaxPool",
    "Softmax",
}

FOLDED_ACTIONS = {
    "Cast": "generation-time dtype/shape helper",
    "Constant": "generation-time constant",
    "DequantizeLinear": "generation-time quantization boundary",
    "Flatten": "generation-time shape fold",
    "Gather": "generation-time shape/index helper",
    "QuantizeLinear": "generation-time quantization boundary",
    "Reshape": "generation-time shape fold",
    "Shape": "generation-time shape helper",
    "Slice": "generation-time shape/slice helper",
    "Unsqueeze": "generation-time shape helper",
}

FUSED_ACTIONS = {
    "Clip": "cmsis_nn_activation min/max when adjacent runtime op is generated",
    "Relu": "cmsis_nn_activation min/max when adjacent runtime op is generated",
}


def map_model(graph: ModelGraph) -> list[OpMapping]:
    return [_map_node(graph, node) for node in graph.nodes]


def _map_node(graph: ModelGraph, node: NodeSpec) -> OpMapping:
    layout_note = _layout_note(graph, node)
    if node.converter_status == "unsupported":
        potential = _potential_action(node)
        if potential is not None:
            status, action, needs_scratch = potential
            return _mapping(
                node,
                status="blocked",
                action=action,
                reason=(
                    "converter marked this op as unsupported; upstream schema support is "
                    "required before codegen can emit CMSIS-NN code"
                ),
                needs_quantization=status in {"direct_api", "wrapper_api"},
                needs_scratch=needs_scratch,
                layout_note=layout_note,
            )
        return _mapping(
            node,
            status="unsupported",
            action="none",
            reason="op is unsupported by converter and has no CMSIS-NN mapping in this version",
            layout_note=layout_note,
        )

    if node.op_type in FOLDED_ACTIONS:
        return _mapping(
            node,
            status="folded",
            action=FOLDED_ACTIONS[node.op_type],
            reason="no runtime CMSIS-NN call is required",
            layout_note=layout_note,
        )

    if node.op_type in FUSED_ACTIONS:
        return _mapping(
            node,
            status="fused",
            action=FUSED_ACTIONS[node.op_type],
            reason="activation should be folded into neighboring CMSIS-NN parameters",
            layout_note=layout_note,
        )

    if node.op_type == "BatchNormalization":
        return _mapping(
            node,
            status="blocked",
            action="fold into Conv/FC parameters",
            reason=(
                "BatchNormalization folding needs weight and bias rewrite; first version reports "
                "the required action instead of emitting unsafe runtime code"
            ),
            layout_note=layout_note,
        )

    if node.op_type == "Conv" and _is_depthwise_conv(node):
        return _mapping(
            node,
            status="blocked",
            action="arm_depthwise_conv_wrapper_s8",
            reason="depthwise/grouped Conv needs a dedicated CMSIS-NN renderer",
            needs_quantization=True,
            needs_scratch=True,
            layout_note=layout_note,
        )
    else:
        base = RUNTIME_ACTIONS.get(node.op_type)

    if base is None:
        return _mapping(
            node,
            status="unsupported",
            action="none",
            reason="no CMSIS-NN mapping is defined for this ONNX op",
            layout_note=layout_note,
        )

    status, action, needs_scratch = base
    if node.op_type == "Conv" and _conv_has_unsupported_shape(node):
        return _mapping(
            node,
            status="blocked",
            action=action,
            reason="first Conv renderer supports groups=1 and dilation=[1, 1] only",
            needs_quantization=True,
            needs_scratch=needs_scratch,
            layout_note=layout_note,
        )
    if not graph.has_quantization:
        return _mapping(
            node,
            status="blocked",
            action=action,
            reason=(
                "CMSIS-NN s8 runtime path requires quantization parameters, but converter output "
                "does not contain a quantization section"
            ),
            needs_quantization=True,
            needs_scratch=needs_scratch,
            layout_note=layout_note,
        )
    if node.op_type not in RENDERED_RUNTIME_OPS:
        return _mapping(
            node,
            status="blocked",
            action=action,
            reason=(
                "quantization is present, but this runtime op does not yet have a real "
                "CMSIS-NN renderer in generator.py"
            ),
            needs_quantization=True,
            needs_scratch=needs_scratch,
            layout_note=layout_note,
        )

    return _mapping(
        node,
        status=status,
        action=action,
        reason="required quantization section is present",
        needs_quantization=True,
        needs_scratch=needs_scratch,
        layout_note=layout_note,
    )


def _potential_action(node: NodeSpec) -> tuple[str, str, bool] | None:
    if node.op_type == "Conv" and _is_depthwise_conv(node):
        return ("wrapper_api", "arm_depthwise_conv_wrapper_s8", True)
    if node.op_type in FOLDED_ACTIONS:
        return ("folded", FOLDED_ACTIONS[node.op_type], False)
    if node.op_type in FUSED_ACTIONS:
        return ("fused", FUSED_ACTIONS[node.op_type], False)
    if node.op_type == "BatchNormalization":
        return ("blocked", "fold into Conv/FC parameters", False)
    return RUNTIME_ACTIONS.get(node.op_type)


def _mapping(
    node: NodeSpec,
    *,
    status: str,
    action: str,
    reason: str,
    needs_quantization: bool = False,
    needs_scratch: bool = False,
    layout_note: str = "",
) -> OpMapping:
    return OpMapping(
        index=node.index,
        node_name=node.name,
        onnx_op=node.op_type,
        converter_status=node.converter_status,
        status=status,
        cmsis_action=action,
        reason=reason,
        inputs=node.inputs,
        outputs=node.outputs,
        input_shapes=node.input_shapes,
        output_shapes=node.output_shapes,
        needs_quantization=needs_quantization,
        needs_scratch=needs_scratch,
        layout_note=layout_note,
    )


def _is_depthwise_conv(node: NodeSpec) -> bool:
    if node.op_type != "Conv":
        return False
    group = int(node.normalized_attributes.get("group", node.attributes.get("group", 1)))
    weight_shape = _weight_shape(node)
    if not weight_shape or len(weight_shape) != 4:
        return group > 1
    out_channels, in_per_group, _, _ = weight_shape
    if not isinstance(out_channels, int) or not isinstance(in_per_group, int):
        return group > 1
    return group > 1 and in_per_group == 1 and out_channels % group == 0


def _conv_has_unsupported_shape(node: NodeSpec) -> bool:
    if node.op_type != "Conv":
        return False
    group = int(node.normalized_attributes.get("group", node.attributes.get("group", 1)))
    dilations = node.normalized_attributes.get(
        "dilations",
        node.attributes.get("dilations", [1, 1]),
    )
    if group != 1:
        return True
    if not isinstance(dilations, list) or len(dilations) not in {1, 2}:
        return True
    return any(int(item) != 1 for item in dilations)


def _weight_shape(node: NodeSpec) -> list[int | str | None] | None:
    for weight_name in node.weights:
        shape = node.input_shapes.get(weight_name)
        if shape:
            return shape
    return None


def _layout_note(graph: ModelGraph, node: NodeSpec) -> str:
    if graph.layout == "NHWC":
        return "converter layout already marked NHWC"
    if graph.layout != "NCHW":
        return f"unknown converter layout {graph.layout!r}; strict generation should reject"
    if node.op_type in {"Conv", "MaxPool", "AveragePool", "GlobalAveragePool"}:
        return (
            "converter layout is NCHW; CMSIS-NN runtime uses NHWC, "
            "so layout handling is required"
        )
    if _has_rank4_output(node):
        return "rank-4 tensor follows converter layout NCHW unless explicitly transformed"
    return "no layout transform required for this node in first-version report"


def _has_rank4_output(node: NodeSpec) -> bool:
    for shape in node.output_shapes.values():
        if element_count_from_shape(shape) is not None and len(shape) == 4:
            return True
        if len(shape) == 4:
            return True
    return False
