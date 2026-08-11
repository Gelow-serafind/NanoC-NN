from __future__ import annotations

from .model import ModelGraph, NodeSpec, OpMapping, element_count_from_shape

RUNTIME_ACTIONS = {
    "Abs": ("direct_api", "generated_c_abs_s8", False),
    "Add": ("direct_api", "arm_elementwise_add_s8", True),
    "AveragePool": ("direct_api", "arm_avgpool_s8", True),
    "Ceil": ("direct_api", "generated_c_ceil_s8", False),
    "Clip": ("direct_api", "generated_c_clip_s8", False),
    "Concat": ("direct_api", "arm_concatenation_s8_x/y/z/w", False),
    "Conv": ("wrapper_api", "arm_convolve_wrapper_s8", True),
    "Div": ("direct_api", "generated_c_div_s8", False),
    "Equal": ("direct_api", "generated_c_equal_bool", False),
    "Exp": ("direct_api", "generated_c_exp_s8", False),
    "Floor": ("direct_api", "generated_c_floor_s8", False),
    "Gather": ("direct_api", "generated_c_gather_s8", False),
    "Gemm": ("wrapper_api", "arm_fully_connected_wrapper_s8", True),
    "GlobalAveragePool": ("direct_api", "arm_avgpool_s8", True),
    "Greater": ("direct_api", "generated_c_greater_bool", False),
    "LeakyRelu": ("direct_api", "generated_c_leakyrelu_s8", False),
    "Less": ("direct_api", "generated_c_less_bool", False),
    "Log": ("direct_api", "generated_c_log_s8", False),
    "MatMul": ("wrapper_api", "arm_fully_connected_wrapper_s8", True),
    "Max": ("direct_api", "generated_c_max_s8", False),
    "MaxPool": ("direct_api", "arm_max_pool_s8", True),
    "Min": ("direct_api", "generated_c_min_s8", False),
    "Mul": ("direct_api", "arm_elementwise_mul_s8", True),
    "Neg": ("direct_api", "generated_c_neg_s8", False),
    "Pad": ("direct_api", "generated_c_pad_s8", False),
    "Pow": ("direct_api", "generated_c_pow_s8", False),
    "QLinearAdd": ("direct_api", "arm_elementwise_add_s8", True),
    "QLinearConv": ("wrapper_api", "arm_convolve_wrapper_s8", True),
    "QLinearGlobalAveragePool": ("direct_api", "arm_avgpool_s8", True),
    "QLinearMatMul": ("wrapper_api", "arm_fully_connected_wrapper_s8", True),
    "Reciprocal": ("direct_api", "generated_c_reciprocal_s8", False),
    "ReduceMax": ("direct_api", "generated_c_reducemax_s8", False),
    "ReduceMean": ("direct_api", "generated_c_reducemean_s8", False),
    "ReduceMin": ("direct_api", "generated_c_reducemin_s8", False),
    "ReduceL1": ("direct_api", "generated_c_reducel1_s8", False),
    "ReduceL2": ("direct_api", "generated_c_reducel2_s8", False),
    "ReduceProd": ("direct_api", "generated_c_reduceprod_s8", False),
    "ReduceSum": ("direct_api", "generated_c_reducesum_s8", False),
    "Round": ("direct_api", "generated_c_round_s8", False),
    "Sign": ("direct_api", "generated_c_sign_s8", False),
    "Softmax": ("direct_api", "arm_softmax_s8", True),
    "Sigmoid": ("direct_api", "generated_c_sigmoid_s8", False),
    "Slice": ("direct_api", "generated_c_slice_s8", False),
    "Sqrt": ("direct_api", "generated_c_sqrt_s8", False),
    "Sub": ("direct_api", "generated_c_sub_s8", False),
    "Tanh": ("direct_api", "generated_c_tanh_s8", False),
    "Transpose": ("direct_api", "arm_transpose_s8", True),
    "Unsqueeze": ("direct_api", "generated_c_unsqueeze_s8", False),
    "Where": ("direct_api", "generated_c_where_s8", False),
}

RENDERED_RUNTIME_OPS = {
    "Abs",
    "Add",
    "AveragePool",
    "Ceil",
    "Clip",
    "Concat",
    "Conv",
    "Div",
    "Equal",
    "Exp",
    "Floor",
    "Gather",
    "Gemm",
    "GlobalAveragePool",
    "Greater",
    "LeakyRelu",
    "Less",
    "Log",
    "MatMul",
    "Max",
    "MaxPool",
    "Min",
    "Mul",
    "Neg",
    "Pad",
    "Pow",
    "QLinearAdd",
    "QLinearConv",
    "QLinearGlobalAveragePool",
    "QLinearMatMul",
    "Reciprocal",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceL1",
    "ReduceL2",
    "ReduceProd",
    "ReduceSum",
    "Round",
    "Sign",
    "Softmax",
    "Sigmoid",
    "Slice",
    "Sqrt",
    "Sub",
    "Tanh",
    "Transpose",
    "Unsqueeze",
    "Where",
}

FOLDED_ACTIONS = {
    "Cast": "generation-time dtype/shape helper",
    "Constant": "generation-time constant",
    "DequantizeLinear": "generation-time quantization boundary",
    "Dropout": "generation-time inference no-op",
    "Flatten": "generation-time shape fold",
    "Gather": "generation-time shape/index helper",
    "QuantizeLinear": "generation-time quantization boundary",
    "Reshape": "generation-time shape fold",
    "Squeeze": "generation-time shape fold",
    "Shape": "generation-time shape helper",
    "Slice": "generation-time shape/slice helper",
    "Unsqueeze": "generation-time shape helper",
}

FUSED_ACTIONS = {
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

    if node.op_type in {"Gather", "Slice"} and _is_shape_helper_slice_or_gather(node):
        return _mapping(
            node,
            status="folded",
            action=FOLDED_ACTIONS[node.op_type],
            reason="shape-only helper feeds tensor shape/index construction",
            layout_note=layout_note,
        )

    if node.op_type == "Unsqueeze" and _is_shape_helper_unsqueeze(node):
        return _mapping(
            node,
            status="folded",
            action=FOLDED_ACTIONS[node.op_type],
            reason="shape-only helper feeds tensor shape/index construction",
            layout_note=layout_note,
        )

    if node.op_type in FOLDED_ACTIONS and node.op_type not in {"Gather", "Slice", "Unsqueeze"}:
        return _mapping(
            node,
            status="folded",
            action=FOLDED_ACTIONS[node.op_type],
            reason="no runtime CMSIS-NN call is required",
            layout_note=layout_note,
        )

    if node.op_type == "Concat" and _is_shape_helper_concat(node):
        return _mapping(
            node,
            status="folded",
            action="generation-time shape concat",
            reason="shape-only Concat feeds tensor shape construction",
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

    if node.op_type in {"Conv", "QLinearConv"} and _is_depthwise_conv(node):
        return _mapping(
            node,
            status="wrapper_api",
            action="arm_depthwise_conv_wrapper_s8",
            reason="required quantization section is present",
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
    if node.op_type == "Concat":
        return _mapping(
            node,
            status=status,
            action=action,
            reason="int8 concat copies already-quantized tensor bytes without arithmetic",
            needs_quantization=False,
            needs_scratch=needs_scratch,
            layout_note=layout_note,
        )
    if node.op_type in {"Conv", "QLinearConv"} and _conv_has_unsupported_shape(node):
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
    if node.op_type in {"Conv", "QLinearConv"} and _is_depthwise_conv(node):
        return ("wrapper_api", "arm_depthwise_conv_wrapper_s8", True)
    if node.op_type in {"Gather", "Slice"} and _is_shape_helper_slice_or_gather(node):
        return ("folded", FOLDED_ACTIONS[node.op_type], False)
    if node.op_type == "Unsqueeze" and _is_shape_helper_unsqueeze(node):
        return ("folded", FOLDED_ACTIONS[node.op_type], False)
    if node.op_type in FOLDED_ACTIONS and node.op_type not in {"Gather", "Slice", "Unsqueeze"}:
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
    if node.op_type not in {"Conv", "QLinearConv"}:
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
    if node.op_type not in {"Conv", "QLinearConv"}:
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


def _is_shape_helper_concat(node: NodeSpec) -> bool:
    for shape in node.output_shapes.values():
        if len(shape) <= 1:
            return True
    return False


def _is_shape_helper_slice_or_gather(node: NodeSpec) -> bool:
    if node.op_type not in {"Slice", "Gather"}:
        return False
    shapes = [
        *[shape for shape in node.input_shapes.values()],
        *[shape for shape in node.output_shapes.values()],
    ]
    return bool(shapes) and all(len(shape) <= 1 for shape in shapes)


def _is_shape_helper_unsqueeze(node: NodeSpec) -> bool:
    if node.op_type != "Unsqueeze":
        return False
    shapes = [
        *[shape for shape in node.input_shapes.values()],
        *[shape for shape in node.output_shapes.values()],
    ]
    return bool(shapes) and all(len(shape) <= 1 for shape in shapes)
