"""
tdd/scripts/common/model_builder.py — 通用 ONNX Q/DQ 模型构建工具

提供一系列辅助函数, 用 onnx.helper API 直接构建符合白名单约束的
Q/DQ int8 量化 ONNX 模型, 供各阶段测试脚本复用。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

# ---------------------------------------------------------------------------
# 公共常量
# ---------------------------------------------------------------------------
DEFAULT_OPSET = 17
PRODUCER_NAME = "nanoc-tdd"


# ---------------------------------------------------------------------------
# 量化节点构建
# ---------------------------------------------------------------------------

def make_quantize_linear(
    name: str,
    inputs: list[str],
    outputs: list[str],
    scale: float,
    zero_point: int = 0,
) -> helper.NodeProto:
    """创建一个 QuantizeLinear 节点。

    Args:
        name: 节点名称。
        inputs: [data, scale_tensor, zero_point_tensor]。
        outputs: 输出张量名列表。
        scale: 量化 scale。
        zero_point: 量化 zero_point, 默认 0 (对称量化)。
    """
    return helper.make_node(
        "QuantizeLinear",
        inputs,
        outputs,
        name=name,
    )


def make_dequantize_linear(
    name: str,
    inputs: list[str],
    outputs: list[str],
    scale: float,
    zero_point: int = 0,
) -> helper.NodeProto:
    """创建一个 DequantizeLinear 节点。

    Args:
        name: 节点名称。
        inputs: [data, scale_tensor, zero_point_tensor]。
        outputs: 输出张量名列表。
        scale: 量化 scale。
        zero_point: 量化 zero_point, 默认 0 (对称量化)。
    """
    return helper.make_node(
        "DequantizeLinear",
        inputs,
        outputs,
        name=name,
    )


def make_qdq_wrapper(
    base_name: str,
    tensor_name: str,
    shape: list[int],
    scale: float,
    zero_point: int = 0,
    is_input: bool = True,
) -> tuple[list[helper.NodeProto], list[helper.TensorProto], list[helper.ValueInfoProto]]:
    """为一组输入/输出张量创建完整的 Q/DQ 包装 (含 scale/zp initializer)。

    Args:
        base_name: 基础名称 (用于生成节点和 initializer 名称)。
        tensor_name: 被包装的浮点张量名称。
        shape: 张量形状。
        scale: 量化 scale。
        zero_point: 量化 zero_point。
        is_input: True 表示输入侧包装 (Q→DQ), False 表示输出侧 (Q→DQ)。

    Returns:
        (nodes, initializers, value_infos) 三元组。
    """
    nodes: list[helper.NodeProto] = []
    initializers: list[helper.TensorProto] = []
    value_infos: list[helper.ValueInfoProto] = []

    s_name = f"{base_name}_scale"
    zp_name = f"{base_name}_zp"
    s_init = numpy_helper.from_array(np.array([scale], dtype=np.float32), name=s_name)
    zp_init = numpy_helper.from_array(np.array([zero_point], dtype=np.int8), name=zp_name)
    initializers.extend([s_init, zp_init])

    int_name = f"{base_name}_int8"

    if is_input:
        # 输入: float32 → Q → int8 → DQ → float32 (送入运行期算子)
        nodes.append(
            make_quantize_linear(
                f"{base_name}_Q",
                [tensor_name, s_name, zp_name],
                [int_name],
                scale,
                zero_point,
            )
        )
        nodes.append(
            make_dequantize_linear(
                f"{base_name}_DQ",
                [int_name, s_name, zp_name],
                [f"{tensor_name}_dq"],
                scale,
                zero_point,
            )
        )
        value_infos.append(
            helper.make_tensor_value_info(int_name, TensorProto.INT8, shape)
        )
    else:
        # 输出: float32 ← DQ ← int8 ← Q ← float32 (从运行期算子出来)
        nodes.append(
            make_quantize_linear(
                f"{base_name}_Q",
                [tensor_name, s_name, zp_name],
                [int_name],
                scale,
                zero_point,
            )
        )
        nodes.append(
            make_dequantize_linear(
                f"{base_name}_DQ",
                [int_name, s_name, zp_name],
                [f"{tensor_name}_dq"],
                scale,
                zero_point,
            )
        )
        value_infos.append(
            helper.make_tensor_value_info(int_name, TensorProto.INT8, shape)
        )

    return nodes, initializers, value_infos


# ---------------------------------------------------------------------------
# 算子节点构建
# ---------------------------------------------------------------------------

def make_gemm_node(
    name: str,
    inputs: list[str],
    outputs: list[str],
    weight_shape: list[int],  # [out_features, in_features] for transB=1
    with_bias: bool = True,
    transB: int = 1,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> tuple[list[helper.NodeProto], list[helper.TensorProto]]:
    """创建一个 Gemm 节点及其权重/bias initializer。

    Args:
        name: 节点名称。
        inputs: [input, weight, bias] 或 [input, weight]。
        outputs: 输出张量名列表。
        weight_shape: 权重形状 (transB=1 时为 [O, I])。
        with_bias: 是否包含 bias。
        transB: B 矩阵是否转置, 默认 1。
    """
    initializers: list[helper.TensorProto] = []
    w_q_name = f"{name}.weight.q"
    w_scale_name = f"{name}.weight.scale"
    w_zp_name = f"{name}.weight.zero_point"
    w_dq_name = f"{name}.weight.dq"
    b_name = f"{name}.bias"

    rng = np.random.RandomState(_stable_seed(name))
    weight_arr = rng.randint(-8, 9, size=weight_shape).astype(np.int8)
    initializers.extend(
        [
            numpy_helper.from_array(weight_arr, name=w_q_name),
            numpy_helper.from_array(np.array(0.02, dtype=np.float32), name=w_scale_name),
            numpy_helper.from_array(np.array(0, dtype=np.int8), name=w_zp_name),
        ]
    )

    weight_dq = helper.make_node(
        "DequantizeLinear",
        [w_q_name, w_scale_name, w_zp_name],
        [w_dq_name],
        name=f"{name}_weight_dequant",
    )

    node_inputs = [inputs[0], w_dq_name]
    if with_bias:
        out_features = weight_shape[0]
        bias_arr = (rng.randn(out_features) * 0.01).astype(np.float32)
        initializers.append(numpy_helper.from_array(bias_arr, name=b_name))
        node_inputs.append(b_name)
    else:
        # 无 bias 场景传入空字符串让 ONNX 使用默认 (全零)
        node_inputs.append("")

    node = helper.make_node(
        "Gemm",
        node_inputs,
        outputs,
        name=name,
        transB=transB,
        alpha=alpha,
        beta=beta,
    )
    return [weight_dq, node], initializers


def make_conv_node(
    name: str,
    inputs: list[str],
    outputs: list[str],
    weight_shape: list[int],  # [O, I, H, W] (ONNX OIHW)
    kernel_shape: list[int],
    strides: list[int] | None = None,
    pads: list[int] | None = None,
    group: int = 1,
    dilations: list[int] | None = None,
    with_bias: bool = True,
) -> tuple[list[helper.NodeProto], list[helper.TensorProto]]:
    """创建一个 Conv 节点及其权重/bias initializer。

    Args:
        name: 节点名称。
        inputs: [input, weight, bias] 或 [input, weight]。
        outputs: 输出张量名列表。
        weight_shape: ONNX OIHW 权重形状。
        kernel_shape: [H, W]。
        strides: [H, W], 默认 [1, 1]。
        pads: [top, left, bottom, right], 默认 [0,0,0,0]。
        group: 分组数, 默认 1。
        dilations: [H, W], 默认 [1, 1]。
        with_bias: 是否包含 bias。
    """
    if strides is None:
        strides = [1, 1]
    if pads is None:
        pads = [0, 0, 0, 0]
    if dilations is None:
        dilations = [1, 1]

    initializers: list[helper.TensorProto] = []
    w_q_name = f"{name}.weight.q"
    w_scale_name = f"{name}.weight.scale"
    w_zp_name = f"{name}.weight.zero_point"
    w_dq_name = f"{name}.weight.dq"
    b_name = f"{name}.bias"

    rng = np.random.RandomState(_stable_seed(name))
    weight_arr = rng.randint(-8, 9, size=weight_shape).astype(np.int8)
    initializers.extend(
        [
            numpy_helper.from_array(weight_arr, name=w_q_name),
            numpy_helper.from_array(np.array(0.02, dtype=np.float32), name=w_scale_name),
            numpy_helper.from_array(np.array(0, dtype=np.int8), name=w_zp_name),
        ]
    )

    weight_dq = helper.make_node(
        "DequantizeLinear",
        [w_q_name, w_scale_name, w_zp_name],
        [w_dq_name],
        name=f"{name}_weight_dequant",
    )

    node_inputs = [inputs[0], w_dq_name]
    kwargs: dict = {
        "kernel_shape": kernel_shape,
        "strides": strides,
        "pads": pads,
        "group": group,
        "dilations": dilations,
    }
    if with_bias:
        out_channels = weight_shape[0]
        bias_arr = (rng.randn(out_channels) * 0.01).astype(np.float32)
        initializers.append(numpy_helper.from_array(bias_arr, name=b_name))
        node_inputs.append(b_name)

    node = helper.make_node("Conv", node_inputs, outputs, name=name, **kwargs)
    return [weight_dq, node], initializers


def _stable_seed(name: str) -> int:
    seed = 0
    for char in name:
        seed = (seed * 131 + ord(char)) % (2**31 - 1)
    return seed or 1


def make_maxpool_node(
    name: str,
    inputs: list[str],
    outputs: list[str],
    kernel_shape: list[int],
    strides: list[int] | None = None,
    pads: list[int] | None = None,
) -> helper.NodeProto:
    """创建一个 MaxPool 节点。"""
    if strides is None:
        strides = kernel_shape
    if pads is None:
        pads = [0, 0, 0, 0]
    return helper.make_node(
        "MaxPool",
        inputs,
        outputs,
        name=name,
        kernel_shape=kernel_shape,
        strides=strides,
        pads=pads,
    )


def make_avgpool_node(
    name: str,
    inputs: list[str],
    outputs: list[str],
    kernel_shape: list[int] | None = None,
    strides: list[int] | None = None,
    pads: list[int] | None = None,
) -> helper.NodeProto:
    """创建一个 AveragePool 节点。"""
    kwargs: dict = {}
    if kernel_shape is not None:
        kwargs["kernel_shape"] = kernel_shape
    if strides is not None:
        kwargs["strides"] = strides
    if pads is not None:
        kwargs["pads"] = pads
    return helper.make_node("AveragePool", inputs, outputs, name=name, **kwargs)


def make_global_avgpool_node(
    name: str,
    inputs: list[str],
    outputs: list[str],
) -> helper.NodeProto:
    """创建一个 GlobalAveragePool 节点。"""
    return helper.make_node("GlobalAveragePool", inputs, outputs, name=name)


def make_softmax_node(
    name: str,
    inputs: list[str],
    outputs: list[str],
    axis: int = 1,
) -> helper.NodeProto:
    """创建一个 Softmax 节点。"""
    return helper.make_node("Softmax", inputs, outputs, name=name, axis=axis)


def make_relu_node(
    name: str,
    inputs: list[str],
    outputs: list[str],
) -> helper.NodeProto:
    """创建一个 Relu 节点。"""
    return helper.make_node("Relu", inputs, outputs, name=name)


def make_flatten_node(
    name: str,
    inputs: list[str],
    outputs: list[str],
    axis: int = 1,
) -> helper.NodeProto:
    """创建一个 Flatten 节点。"""
    return helper.make_node("Flatten", inputs, outputs, name=name, axis=axis)


def make_reshape_node(
    name: str,
    inputs: list[str],
    outputs: list[str],
    target_shape: list[int],
) -> tuple[helper.NodeProto, list[helper.TensorProto]]:
    """创建一个 Reshape 节点及其 shape constant initializer。"""
    shape_name = f"{name}.shape"
    shape_const = numpy_helper.from_array(
        np.array(target_shape, dtype=np.int64), name=shape_name
    )
    node = helper.make_node("Reshape", inputs + [shape_name], outputs, name=name)
    return node, [shape_const]


# ---------------------------------------------------------------------------
# 模型组装
# ---------------------------------------------------------------------------

def build_and_save(
    nodes: list[helper.NodeProto],
    inputs: list[helper.ValueInfoProto],
    outputs: list[helper.ValueInfoProto],
    initializers: list[helper.TensorProto],
    save_path: Path,
    graph_name: str = "tdd_test",
    opset: int = DEFAULT_OPSET,
) -> None:
    """组装 ONNX 模型、做 shape inference 并保存。

    Args:
        nodes: 所有节点 (含 Q/DQ + 运行期算子 + 融合算子)。
        inputs: 图输入 value_info。
        outputs: 图输出 value_info。
        initializers: 所有权重 + Q/DQ scale/zp initializer。
        save_path: 输出 .onnx 路径。
        graph_name: 图名称。
        opset: opset 版本。
    """
    graph = helper.make_graph(
        nodes,
        graph_name,
        inputs,
        outputs,
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name=PRODUCER_NAME,
        opset_imports=[helper.make_opsetid("", opset)],
    )

    # 基础合法性检查
    onnx.checker.check_model(model)

    # Shape inference
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass  # 部分模型可能推断失败，保留原始 shape

    save_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, save_path)
