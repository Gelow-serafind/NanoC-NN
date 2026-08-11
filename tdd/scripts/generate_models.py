"""
tdd/scripts/generate_models.py — 根据 cases/ 规格批量生成 ONNX 测试模型

用法:
    python tdd/scripts/generate_models.py
    python tdd/scripts/generate_models.py --case GEMM_001
    python tdd/scripts/generate_models.py --category core/gemm
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from cases_registry import CASE_MAP  # noqa: E402
from common.model_builder import (  # noqa: E402
    build_and_save,
    make_conv_node,
    make_avgpool_node,
    make_flatten_node,
    make_gemm_node,
    make_global_avgpool_node,
    make_maxpool_node,
    make_qdq_wrapper,
    make_relu_node,
    make_reshape_node,
    make_softmax_node,
)

TDD_ROOT = _SCRIPT_DIR.parent
WORK_ROOT = TDD_ROOT / "work"
MODELS_ROOT = WORK_ROOT / "models"
FIXTURES_ROOT = TDD_ROOT / "fixtures"


# ---------------------------------------------------------------------------
# Helper: 构建完整的 Q/DQ 包裹模型
# ---------------------------------------------------------------------------

def _build_qdq_model(
    graph_name: str,
    input_shape: list[int],
    input_scale: float,
    input_zp: int,
    output_shape: list[int],
    output_scale: float,
    output_zp: int,
    runtime_nodes: list,
    runtime_initializers: list,
    save_path: Path,
) -> None:
    """构建一个完整的 Q/DQ 包裹模型并保存。"""
    nodes = []
    initializers = []
    value_infos = []

    input_name = "input"
    output_name = "output"

    inp_vi = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
    out_vi = helper.make_tensor_value_info(f"{output_name}_dq", TensorProto.FLOAT, output_shape)

    in_nodes, in_inits, in_vis = make_qdq_wrapper(
        "input_qdq", input_name, input_shape, input_scale, input_zp, is_input=True
    )
    nodes.extend(in_nodes)
    initializers.extend(in_inits)
    value_infos.extend(in_vis)

    nodes.extend(runtime_nodes)
    initializers.extend(runtime_initializers)

    out_nodes, out_inits, out_vis = make_qdq_wrapper(
        "output_qdq", output_name, output_shape, output_scale, output_zp, is_input=False
    )
    nodes.extend(out_nodes)
    initializers.extend(out_inits)
    value_infos.extend(out_vis)

    build_and_save(
        nodes=nodes,
        inputs=[inp_vi],
        outputs=[out_vi],
        initializers=initializers,
        save_path=save_path,
        graph_name=graph_name,
        value_infos=value_infos,
    )


# ---------------------------------------------------------------------------
# GEMM 用例生成
# ---------------------------------------------------------------------------

def gen_abs_001() -> Path:
    """ABS_001: 官方 Abs QDQ/int8，同量化输入输出。"""
    path = MODELS_ROOT / "core" / "abs" / "ABS_001.onnx"
    abs_node = helper.make_node("Abs", ["input_dq"], ["output"], name="abs")
    _build_qdq_model(
        graph_name="abs_001",
        input_shape=[1, 8],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 8],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[abs_node],
        runtime_initializers=[],
        save_path=path,
    )
    return path


def _gen_unary_simple_case(*, case_id: str, op_type: str, category: str, node_name: str) -> Path:
    """构造单个官方 unary 算子的 QDQ/int8 用例 — [1,8] -> [1,8]。"""
    path = MODELS_ROOT / "core" / category / f"{case_id}.onnx"
    node = helper.make_node(op_type, ["input_dq"], ["output"], name=node_name)
    _build_qdq_model(
        graph_name=f"{case_id.lower()}_qdq",
        input_shape=[1, 8],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 8],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[node],
        runtime_initializers=[],
        save_path=path,
    )
    return path


def gen_erf_001() -> Path:
    """ERF_001: official Erf QDQ — [1,8] -> [1,8]."""
    return _gen_unary_simple_case(case_id="ERF_001", op_type="Erf", category="erf", node_name="erf")


def gen_softplus_001() -> Path:
    """SOFTPLUS_001: official Softplus QDQ — [1,8] -> [1,8]."""
    return _gen_unary_simple_case(
        case_id="SOFTPLUS_001", op_type="Softplus", category="softplus", node_name="softplus"
    )


def gen_softsign_001() -> Path:
    """SOFTSIGN_001: official Softsign QDQ — [1,8] -> [1,8]."""
    return _gen_unary_simple_case(
        case_id="SOFTSIGN_001", op_type="Softsign", category="softsign", node_name="softsign"
    )


def gen_hardswish_001() -> Path:
    """HARDSWISH_001: official HardSwish QDQ — [1,8] -> [1,8]."""
    return _gen_unary_simple_case(
        case_id="HARDSWISH_001", op_type="HardSwish", category="hardswish", node_name="hardswish"
    )


def _gen_unary_attr_case(
    *,
    case_id: str,
    op_type: str,
    category: str,
    node_name: str,
    attrs: dict[str, float],
) -> Path:
    """构造带属性的官方 unary 算子 QDQ/int8 用例 — [1,8] -> [1,8]。"""
    path = MODELS_ROOT / "core" / category / f"{case_id}.onnx"
    node = helper.make_node(
        op_type, ["input_dq"], ["output"], name=node_name, **{k: float(v) for k, v in attrs.items()}
    )
    _build_qdq_model(
        graph_name=f"{case_id.lower()}_qdq",
        input_shape=[1, 8],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 8],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[node],
        runtime_initializers=[],
        save_path=path,
    )
    return path


def gen_elu_001() -> Path:
    """ELU_001: official Elu alpha=0.5 QDQ — [1,8] -> [1,8]."""
    return _gen_unary_attr_case(
        case_id="ELU_001", op_type="Elu", category="elu", node_name="elu", attrs={"alpha": 0.5}
    )


def gen_selu_001() -> Path:
    """SELU_001: official Selu QDQ — [1,8] -> [1,8]."""
    return _gen_unary_attr_case(
        case_id="SELU_001",
        op_type="Selu",
        category="selu",
        node_name="selu",
        attrs={"alpha": 1.67326, "gamma": 1.0507},
    )


def gen_hardsigmoid_001() -> Path:
    """HARDSIGMOID_001: official HardSigmoid QDQ — [1,8] -> [1,8]."""
    return _gen_unary_attr_case(
        case_id="HARDSIGMOID_001",
        op_type="HardSigmoid",
        category="hardsigmoid",
        node_name="hardsigmoid",
        attrs={"alpha": 0.2, "beta": 0.5},
    )


def gen_thresholdedrelu_001() -> Path:
    """THRESHOLDEDRELU_001: official ThresholdedRelu QDQ — [1,8] -> [1,8]."""
    return _gen_unary_attr_case(
        case_id="THRESHOLDEDRELU_001",
        op_type="ThresholdedRelu",
        category="thresholdedrelu",
        node_name="thresholdedrelu",
        attrs={"alpha": 1.0},
    )


def gen_celu_001() -> Path:
    """CELU_001: official Celu QDQ — [1,8] -> [1,8]."""
    return _gen_unary_attr_case(
        case_id="CELU_001", op_type="Celu", category="celu", node_name="celu", attrs={"alpha": 1.0}
    )


def gen_prelu_001() -> Path:
    """PRELU_001: official PRelu with constant slope — [1,8] -> [1,8]."""
    path = MODELS_ROOT / "core" / "prelu" / "PRELU_001.onnx"
    slope_nodes, slope_inits, _slope_vis, slope_dq = _make_qdq_constant(
        prefix="prelu.slope",
        values=np.array([10, 5, 20, 0, 15, 10, 5, 20], dtype=np.int8).reshape(1, 8),
        scale=0.05,
        zero_point=0,
    )
    prelu_node = helper.make_node(
        "PRelu",
        ["input_dq", slope_dq],
        ["output"],
        name="prelu",
    )
    _build_qdq_model(
        graph_name="prelu_001_qdq_const",
        input_shape=[1, 8],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 8],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=slope_nodes + [prelu_node],
        runtime_initializers=slope_inits,
        save_path=path,
    )
    return path


def gen_gemm_001() -> Path:
    """GEMM_001: 最小对称 FC 无 bias — [1,2] → [1,3]"""
    path = MODELS_ROOT / "core" / "gemm" / "GEMM_001.onnx"
    gemm_nodes, gemm_inits = make_gemm_node(
        "gemm", ["input_dq"], ["output"], weight_shape=[3, 2], with_bias=False
    )
    _build_qdq_model(
        graph_name="gemm_001",
        input_shape=[1, 2], input_scale=0.01, input_zp=0,
        output_shape=[1, 3], output_scale=0.05, output_zp=0,
        runtime_nodes=gemm_nodes,
        runtime_initializers=gemm_inits,
        save_path=path,
    )
    return path


def _copy_fixture_model(case_id: str, fixture_name: str) -> Path:
    path = MODELS_ROOT / "networks" / f"{case_id}.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXTURES_ROOT / "onnx" / fixture_name, path)
    return path


def gen_gemm_002() -> Path:
    """GEMM_002: 非对称输入 zp=10 含 bias — [1,4] → [1,8]"""
    path = MODELS_ROOT / "core" / "gemm" / "GEMM_002.onnx"
    gemm_nodes, gemm_inits = make_gemm_node(
        "gemm", ["input_dq"], ["output"], weight_shape=[8, 4], with_bias=True
    )
    _build_qdq_model(
        graph_name="gemm_002",
        input_shape=[1, 4], input_scale=0.01, input_zp=10,
        output_shape=[1, 8], output_scale=0.05, output_zp=0,
        runtime_nodes=gemm_nodes,
        runtime_initializers=gemm_inits,
        save_path=path,
    )
    return path


def gen_gemm_003() -> Path:
    """GEMM_003: 非 4 对齐维度 13→7"""
    path = MODELS_ROOT / "core" / "gemm" / "GEMM_003.onnx"
    gemm_nodes, gemm_inits = make_gemm_node(
        "gemm", ["input_dq"], ["output"], weight_shape=[7, 13], with_bias=True
    )
    _build_qdq_model(
        graph_name="gemm_003",
        input_shape=[1, 13], input_scale=0.03, input_zp=0,
        output_shape=[1, 7], output_scale=0.04, output_zp=0,
        runtime_nodes=gemm_nodes,
        runtime_initializers=gemm_inits,
        save_path=path,
    )
    return path


def gen_gemm_004() -> Path:
    """GEMM_004: 中等规模 64→32"""
    path = MODELS_ROOT / "core" / "gemm" / "GEMM_004.onnx"
    gemm_nodes, gemm_inits = make_gemm_node(
        "gemm", ["input_dq"], ["output"], weight_shape=[32, 64], with_bias=True
    )
    _build_qdq_model(
        graph_name="gemm_004",
        input_shape=[1, 64], input_scale=0.01, input_zp=0,
        output_shape=[1, 32], output_scale=0.02, output_zp=0,
        runtime_nodes=gemm_nodes,
        runtime_initializers=gemm_inits,
        save_path=path,
    )
    return path


def gen_matmul_001() -> Path:
    """MATMUL_001: 官方 MatMul FC-compatible — [1,4] x [4,3] -> [1,3]."""
    path = MODELS_ROOT / "core" / "matmul" / "MATMUL_001.onnx"
    weight_q = np.array(
        [
            [3, -2, 1],
            [1, 4, -3],
            [-2, 1, 5],
            [2, -1, 3],
        ],
        dtype=np.int8,
    )
    runtime_nodes = [
        helper.make_node(
            "DequantizeLinear",
            ["matmul.weight.q", "matmul.weight.scale", "matmul.weight.zero_point"],
            ["matmul.weight.dq"],
            name="matmul_weight_dequant",
        ),
        helper.make_node(
            "MatMul",
            ["input_dq", "matmul.weight.dq"],
            ["output"],
            name="matmul",
        ),
    ]
    runtime_initializers = [
        numpy_helper.from_array(weight_q, name="matmul.weight.q"),
        numpy_helper.from_array(np.array(0.04, dtype=np.float32), name="matmul.weight.scale"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), name="matmul.weight.zero_point"),
    ]
    _build_qdq_model(
        graph_name="matmul_001",
        input_shape=[1, 4],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 3],
        output_scale=0.04,
        output_zp=0,
        runtime_nodes=runtime_nodes,
        runtime_initializers=runtime_initializers,
        save_path=path,
    )
    return path


# ---------------------------------------------------------------------------
# CONV 用例生成
# ---------------------------------------------------------------------------

def gen_flatten_001() -> Path:
    """FLATTEN_001: official Flatten QDQ axis=1 — [1,2,2,3] → [1,12]"""
    path = MODELS_ROOT / "core" / "flatten" / "FLATTEN_001.onnx"
    flatten_node = make_flatten_node(
        "flatten",
        ["input_dq"],
        ["output"],
        axis=1,
    )
    _build_qdq_model(
        graph_name="flatten_001_axis1_qdq",
        input_shape=[1, 2, 2, 3],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 12],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[flatten_node],
        runtime_initializers=[],
        save_path=path,
    )
    return path


def gen_reshape_001() -> Path:
    """RESHAPE_001: official Reshape QDQ — [1,2,3] -> [1,3,2]."""
    path = MODELS_ROOT / "core" / "reshape" / "RESHAPE_001.onnx"
    reshape_node, reshape_inits = make_reshape_node(
        "reshape",
        ["input_dq"],
        ["output"],
        target_shape=[1, 3, 2],
    )
    _build_qdq_model(
        graph_name="reshape_001_qdq",
        input_shape=[1, 2, 3],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 3, 2],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[reshape_node],
        runtime_initializers=reshape_inits,
        save_path=path,
    )
    return path


def gen_squeeze_001() -> Path:
    """SQUEEZE_001: official Squeeze QDQ — [1,1,2,3] -> [1,2,3]."""
    path = MODELS_ROOT / "core" / "squeeze" / "SQUEEZE_001.onnx"
    axes_name = "squeeze.axes"
    axes_init = numpy_helper.from_array(np.array([1], dtype=np.int64), name=axes_name)
    squeeze_node = helper.make_node(
        "Squeeze",
        ["input_dq", axes_name],
        ["output"],
        name="squeeze",
    )
    _build_qdq_model(
        graph_name="squeeze_001_qdq",
        input_shape=[1, 1, 2, 3],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 2, 3],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[squeeze_node],
        runtime_initializers=[axes_init],
        save_path=path,
    )
    return path


# ---------------------------------------------------------------------------
# CONV 用例生成
# ---------------------------------------------------------------------------

def gen_conv_001() -> Path:
    """CONV_001: 1×1 pointwise 单通道 — [1,1,4,4] → [1,1,4,4]"""
    path = MODELS_ROOT / "core" / "conv" / "CONV_001.onnx"
    conv_nodes, conv_inits = make_conv_node(
        "conv", ["input_dq"], ["output"],
        weight_shape=[1, 1, 1, 1], kernel_shape=[1, 1],
        strides=[1, 1], pads=[0, 0, 0, 0], with_bias=True,
    )
    _build_qdq_model(
        graph_name="conv_001",
        input_shape=[1, 1, 4, 4], input_scale=0.01, input_zp=0,
        output_shape=[1, 1, 4, 4], output_scale=0.05, output_zp=0,
        runtime_nodes=conv_nodes,
        runtime_initializers=conv_inits,
        save_path=path,
    )
    return path


def gen_conv_002() -> Path:
    """CONV_002: 3×3 标准 SAME padding — [1,3,8,8] → [1,8,8,8]"""
    path = MODELS_ROOT / "core" / "conv" / "CONV_002.onnx"
    conv_nodes, conv_inits = make_conv_node(
        "conv", ["input_dq"], ["output"],
        weight_shape=[8, 3, 3, 3], kernel_shape=[3, 3],
        strides=[1, 1], pads=[1, 1, 1, 1], with_bias=True,
    )
    _build_qdq_model(
        graph_name="conv_002",
        input_shape=[1, 3, 8, 8], input_scale=0.01, input_zp=0,
        output_shape=[1, 8, 8, 8], output_scale=0.05, output_zp=0,
        runtime_nodes=conv_nodes,
        runtime_initializers=conv_inits,
        save_path=path,
    )
    return path


def gen_conv_003() -> Path:
    """CONV_003: stride=2 下采样 — [1,4,8,8] → [1,8,4,4]"""
    path = MODELS_ROOT / "core" / "conv" / "CONV_003.onnx"
    conv_nodes, conv_inits = make_conv_node(
        "conv", ["input_dq"], ["output"],
        weight_shape=[8, 4, 3, 3], kernel_shape=[3, 3],
        strides=[2, 2], pads=[1, 1, 1, 1], with_bias=True,
    )
    _build_qdq_model(
        graph_name="conv_003",
        input_shape=[1, 4, 8, 8], input_scale=0.01, input_zp=0,
        output_shape=[1, 8, 4, 4], output_scale=0.03, output_zp=0,
        runtime_nodes=conv_nodes,
        runtime_initializers=conv_inits,
        save_path=path,
    )
    return path


def gen_conv_004() -> Path:
    """CONV_004: 非对称输入 zp=12 — [1,3,8,8] → [1,8,8,8]"""
    path = MODELS_ROOT / "core" / "conv" / "CONV_004.onnx"
    conv_nodes, conv_inits = make_conv_node(
        "conv", ["input_dq"], ["output"],
        weight_shape=[8, 3, 3, 3], kernel_shape=[3, 3],
        strides=[1, 1], pads=[1, 1, 1, 1], with_bias=True,
    )
    _build_qdq_model(
        graph_name="conv_004",
        input_shape=[1, 3, 8, 8], input_scale=0.01, input_zp=12,
        output_shape=[1, 8, 8, 8], output_scale=0.05, output_zp=0,
        runtime_nodes=conv_nodes,
        runtime_initializers=conv_inits,
        save_path=path,
    )
    return path


# ---------------------------------------------------------------------------
# MAXPOOL 用例生成
# ---------------------------------------------------------------------------

def gen_maxpool_001() -> Path:
    """MAXPOOL_001: 标准 2×2 stride=2 — [1,4,8,8] → [1,4,4,4]"""
    path = MODELS_ROOT / "core" / "maxpool" / "MAXPOOL_001.onnx"
    pool_node = make_maxpool_node(
        "maxpool", ["input_dq"], ["output"],
        kernel_shape=[2, 2], strides=[2, 2], pads=[0, 0, 0, 0],
    )
    _build_qdq_model(
        graph_name="maxpool_001",
        input_shape=[1, 4, 8, 8], input_scale=0.05, input_zp=0,
        output_shape=[1, 4, 4, 4], output_scale=0.05, output_zp=0,
        runtime_nodes=[pool_node],
        runtime_initializers=[],
        save_path=path,
    )
    return path


def gen_maxpool_002() -> Path:
    """MAXPOOL_002: DQ→MaxPool→Q 边界 — [1,4,4,4] → [1,4,2,2]"""
    path = MODELS_ROOT / "core" / "maxpool" / "MAXPOOL_002.onnx"
    input_name = "input"
    input_shape = [1, 4, 4, 4]
    output_shape = [1, 4, 2, 2]
    scale = 0.1
    zero_point = 0

    nodes = []
    initializers = []
    value_infos = []

    inp_vi = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
    out_vi = helper.make_tensor_value_info("pool_out_dq", TensorProto.FLOAT, output_shape)

    in_nodes, in_inits, in_vis = make_qdq_wrapper(
        "input_qdq",
        input_name,
        input_shape,
        scale,
        zero_point,
        is_input=True,
    )
    nodes.extend(in_nodes)
    initializers.extend(in_inits)
    value_infos.extend(in_vis)

    pool_node = make_maxpool_node(
        "maxpool",
        ["input_dq"],
        ["pool_out"],
        kernel_shape=[2, 2],
        strides=[2, 2],
        pads=[0, 0, 0, 0],
    )
    nodes.append(pool_node)
    value_infos.append(helper.make_tensor_value_info("pool_out", TensorProto.FLOAT, output_shape))

    pool_q_nodes, pool_q_inits, pool_q_vis = make_qdq_wrapper(
        "pool_out_qdq",
        "pool_out",
        output_shape,
        scale,
        zero_point,
        is_input=False,
    )
    nodes.extend(pool_q_nodes)
    initializers.extend(pool_q_inits)
    value_infos.extend(pool_q_vis)

    build_and_save(
        nodes=nodes,
        inputs=[inp_vi],
        outputs=[out_vi],
        initializers=initializers,
        save_path=path,
        graph_name="maxpool_002_dq_q_boundary",
        value_infos=value_infos,
    )
    return path


# ---------------------------------------------------------------------------
# AVGPOOL 用例生成
# ---------------------------------------------------------------------------

def gen_avgpool_001() -> Path:
    """AVGPOOL_001: QLinearGlobalAveragePool — [1,4,3,3] → [1,4,1,1]"""
    path = MODELS_ROOT / "core" / "avgpool" / "AVGPOOL_001.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)

    input_name = "input"
    output_name = "output"
    input_shape = [1, 4, 3, 3]
    output_shape = [1, 4, 1, 1]

    scale_init = numpy_helper.from_array(np.array([0.05], dtype=np.float32), name="scale")
    zp_init = numpy_helper.from_array(np.array([128], dtype=np.uint8), name="zp")
    initializers = [scale_init, zp_init]

    nodes = [
        helper.make_node(
            "QuantizeLinear",
            [input_name, "scale", "zp"],
            ["input_q"],
            name="input_quant",
        ),
        helper.make_node(
            "QLinearGlobalAveragePool",
            ["input_q", "scale", "zp", "scale", "zp"],
            ["pool_out"],
            name="global_avgpool",
            domain="com.microsoft",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["pool_out", "scale", "zp"],
            [output_name],
            name="output_dequant",
        ),
    ]

    value_infos = [
        helper.make_tensor_value_info("input_q", TensorProto.UINT8, input_shape),
        helper.make_tensor_value_info("pool_out", TensorProto.UINT8, output_shape),
    ]
    inp_vi = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
    out_vi = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)

    graph = helper.make_graph(
        nodes,
        "avgpool_001_qlinear_global",
        [inp_vi],
        [out_vi],
        initializer=initializers,
        value_info=value_infos,
    )
    model = helper.make_model(
        graph,
        producer_name="nanoc-tdd",
        opset_imports=[
            helper.make_opsetid("", 12),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    return path


def gen_avgpool_002() -> Path:
    """AVGPOOL_002: official GlobalAveragePool QDQ — [1,4,3,3] → [1,4,1,1]"""
    path = MODELS_ROOT / "core" / "avgpool" / "AVGPOOL_002.onnx"
    pool_node = make_global_avgpool_node(
        "global_avgpool",
        ["input_dq"],
        ["output"],
    )
    _build_qdq_model(
        graph_name="avgpool_002_official_global",
        input_shape=[1, 4, 3, 3],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 4, 1, 1],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[pool_node],
        runtime_initializers=[],
        save_path=path,
    )
    return path


def gen_avgpool_003() -> Path:
    """AVGPOOL_003: official AveragePool QDQ — [1,2,4,4] -> [1,2,2,2]."""
    path = MODELS_ROOT / "core" / "avgpool" / "AVGPOOL_003.onnx"
    input_shape = [1, 2, 4, 4]
    output_shape = [1, 2, 2, 2]
    scale = 0.05
    zero_point = 0
    input_name = "input"
    output_name = "output"
    pool_node = make_avgpool_node(
        "avgpool",
        ["input_dq"],
        ["output"],
        kernel_shape=[2, 2],
        strides=[2, 2],
        pads=[0, 0, 0, 0],
    )
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            [input_name, "avgpool_003_scale", "avgpool_003_zp"],
            ["input_q"],
            name="input_quant",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["input_q", "avgpool_003_scale", "avgpool_003_zp"],
            ["input_dq"],
            name="input_dequant",
        ),
        pool_node,
        helper.make_node(
            "QuantizeLinear",
            [output_name, "avgpool_003_scale", "avgpool_003_zp"],
            ["output_q"],
            name="output_quant",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["output_q", "avgpool_003_scale", "avgpool_003_zp"],
            ["output_dq"],
            name="output_dequant",
        ),
    ]
    initializers = [
        numpy_helper.from_array(np.array(scale, dtype=np.float32), name="avgpool_003_scale"),
        numpy_helper.from_array(np.array(zero_point, dtype=np.int8), name="avgpool_003_zp"),
    ]
    value_infos = [
        helper.make_tensor_value_info("input_q", TensorProto.INT8, input_shape),
        helper.make_tensor_value_info("input_dq", TensorProto.FLOAT, input_shape),
        helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape),
        helper.make_tensor_value_info("output_q", TensorProto.INT8, output_shape),
    ]
    build_and_save(
        nodes=nodes,
        inputs=[helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)],
        outputs=[helper.make_tensor_value_info("output_dq", TensorProto.FLOAT, output_shape)],
        initializers=initializers,
        save_path=path,
        graph_name="avgpool_003_official_window",
        value_infos=value_infos,
    )
    return path


# ---------------------------------------------------------------------------
# SOFTMAX 用例生成
# ---------------------------------------------------------------------------

def gen_softmax_001() -> Path:
    """SOFTMAX_001: 10 分类标准 — [1,10] → [1,10]"""
    path = MODELS_ROOT / "core" / "softmax" / "SOFTMAX_001.onnx"
    softmax_node = make_softmax_node(
        "softmax", ["input_dq"], ["output"], axis=1
    )
    _build_qdq_model(
        graph_name="softmax_001",
        input_shape=[1, 10], input_scale=0.05, input_zp=0,
        output_shape=[1, 10], output_scale=1.0 / 256, output_zp=-128,
        runtime_nodes=[softmax_node],
        runtime_initializers=[],
        save_path=path,
    )
    return path


def gen_softmax_002() -> Path:
    """SOFTMAX_002: 末端 float output Softmax — [1,4] → [1,4]"""
    path = MODELS_ROOT / "core" / "softmax" / "SOFTMAX_002.onnx"
    input_name = "input"
    input_shape = [1, 4]
    output_shape = [1, 4]

    nodes = []
    initializers = []
    value_infos = []

    inp_vi = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
    out_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

    in_nodes, in_inits, in_vis = make_qdq_wrapper(
        "input_qdq",
        input_name,
        input_shape,
        0.1,
        0,
        is_input=True,
    )
    nodes.extend(in_nodes)
    initializers.extend(in_inits)
    value_infos.extend(in_vis)
    nodes.append(make_softmax_node("softmax", ["input_dq"], ["output"], axis=1))

    build_and_save(
        nodes=nodes,
        inputs=[inp_vi],
        outputs=[out_vi],
        initializers=initializers,
        save_path=path,
        graph_name="softmax_002_float_output",
        value_infos=value_infos,
    )
    return path


# ---------------------------------------------------------------------------
# CONCAT 用例生成
# ---------------------------------------------------------------------------

def gen_concat_001() -> Path:
    """CONCAT_001: 双输入 channel 维 Concat — [1,1,2,2] + [1,1,2,2] → [1,2,2,2]"""
    path = MODELS_ROOT / "core" / "concat" / "CONCAT_001.onnx"
    input_a = "input_a"
    input_b = "input_b"
    input_shape = [1, 1, 2, 2]
    output_shape = [1, 2, 2, 2]
    scale = 0.1
    zero_point = 0

    nodes = []
    initializers = []
    value_infos = []

    inputs = [
        helper.make_tensor_value_info(input_a, TensorProto.FLOAT, input_shape),
        helper.make_tensor_value_info(input_b, TensorProto.FLOAT, input_shape),
    ]
    output = helper.make_tensor_value_info("concat_out_dq", TensorProto.FLOAT, output_shape)

    for tensor_name in (input_a, input_b):
        q_nodes, q_inits, q_vis = make_qdq_wrapper(
            f"{tensor_name}_qdq",
            tensor_name,
            input_shape,
            scale,
            zero_point,
            is_input=True,
        )
        nodes.extend(q_nodes)
        initializers.extend(q_inits)
        value_infos.extend(q_vis)
        value_infos.append(
            helper.make_tensor_value_info(f"{tensor_name}_dq", TensorProto.FLOAT, input_shape)
        )

    nodes.append(
        helper.make_node(
            "Concat",
            ["input_a_dq", "input_b_dq"],
            ["concat_out"],
            name="concat",
            axis=1,
        )
    )
    value_infos.append(
        helper.make_tensor_value_info("concat_out", TensorProto.FLOAT, output_shape)
    )

    out_nodes, out_inits, out_vis = make_qdq_wrapper(
        "output_qdq",
        "concat_out",
        output_shape,
        scale,
        zero_point,
        is_input=False,
    )
    nodes.extend(out_nodes)
    initializers.extend(out_inits)
    value_infos.extend(out_vis)

    build_and_save(
        nodes=nodes,
        inputs=inputs,
        outputs=[output],
        initializers=initializers,
        save_path=path,
        graph_name="concat_001",
        value_infos=value_infos,
    )
    return path


def gen_concat_002() -> Path:
    """CONCAT_002: 不同输入 scale 的 channel 维 Concat — [1,1,2,2] + [1,1,2,2] → [1,2,2,2]"""
    path = MODELS_ROOT / "core" / "concat" / "CONCAT_002.onnx"
    input_a = "input_a"
    input_b = "input_b"
    input_shape = [1, 1, 2, 2]
    output_shape = [1, 2, 2, 2]

    nodes = []
    initializers = []
    value_infos = []
    inputs = [
        helper.make_tensor_value_info(input_a, TensorProto.FLOAT, input_shape),
        helper.make_tensor_value_info(input_b, TensorProto.FLOAT, input_shape),
    ]
    output = helper.make_tensor_value_info("concat_out_dq", TensorProto.FLOAT, output_shape)

    for tensor_name, scale in ((input_a, 0.1), (input_b, 0.2)):
        q_nodes, q_inits, q_vis = make_qdq_wrapper(
            f"{tensor_name}_qdq",
            tensor_name,
            input_shape,
            scale,
            0,
            is_input=True,
        )
        nodes.extend(q_nodes)
        initializers.extend(q_inits)
        value_infos.extend(q_vis)
        value_infos.append(
            helper.make_tensor_value_info(f"{tensor_name}_dq", TensorProto.FLOAT, input_shape)
        )

    nodes.append(
        helper.make_node(
            "Concat",
            ["input_a_dq", "input_b_dq"],
            ["concat_out"],
            name="concat",
            axis=1,
        )
    )
    value_infos.append(
        helper.make_tensor_value_info("concat_out", TensorProto.FLOAT, output_shape)
    )

    out_nodes, out_inits, out_vis = make_qdq_wrapper(
        "output_qdq",
        "concat_out",
        output_shape,
        0.2,
        0,
        is_input=False,
    )
    nodes.extend(out_nodes)
    initializers.extend(out_inits)
    value_infos.extend(out_vis)

    build_and_save(
        nodes=nodes,
        inputs=inputs,
        outputs=[output],
        initializers=initializers,
        save_path=path,
        graph_name="concat_002_requant",
        value_infos=value_infos,
    )
    return path


def _make_qdq_constant(
    *,
    prefix: str,
    values: np.ndarray,
    scale: float,
    zero_point: int,
) -> tuple[list, list, list, str]:
    q_name = f"{prefix}.q"
    scale_name = f"{prefix}.scale"
    zp_name = f"{prefix}.zero_point"
    dq_name = f"{prefix}.dq"
    initializers = [
        numpy_helper.from_array(values.astype(np.int8), name=q_name),
        numpy_helper.from_array(np.array(scale, dtype=np.float32), name=scale_name),
        numpy_helper.from_array(np.array(zero_point, dtype=np.int8), name=zp_name),
    ]
    nodes = [
        helper.make_node(
            "DequantizeLinear",
            [q_name, scale_name, zp_name],
            [dq_name],
            name=f"{prefix}_dequant",
        )
    ]
    value_infos = [helper.make_tensor_value_info(dq_name, TensorProto.FLOAT, list(values.shape))]
    return nodes, initializers, value_infos, dq_name


def gen_add_001() -> Path:
    """ADD_001: official Add QDQ with constant second input — [1,8] -> [1,8]."""
    path = MODELS_ROOT / "core" / "add" / "ADD_001.onnx"
    const_nodes, const_inits, const_vis, const_dq = _make_qdq_constant(
        prefix="add.const",
        values=np.array([2, -1, 3, -2, 1, -3, 2, 0], dtype=np.int8).reshape(1, 8),
        scale=0.05,
        zero_point=0,
    )
    add_node = helper.make_node(
        "Add",
        ["input_dq", const_dq],
        ["output"],
        name="add",
    )
    _build_qdq_model(
        graph_name="add_001_qdq_const",
        input_shape=[1, 8],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 8],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=const_nodes + [add_node],
        runtime_initializers=const_inits,
        save_path=path,
    )
    return path


def gen_mul_001() -> Path:
    """MUL_001: official Mul QDQ with constant second input — [1,8] -> [1,8]."""
    path = MODELS_ROOT / "core" / "mul" / "MUL_001.onnx"
    const_nodes, const_inits, const_vis, const_dq = _make_qdq_constant(
        prefix="mul.const",
        values=np.array([2, -1, 1, 3, -2, 2, 1, -1], dtype=np.int8).reshape(1, 8),
        scale=0.05,
        zero_point=0,
    )
    mul_node = helper.make_node(
        "Mul",
        ["input_dq", const_dq],
        ["output"],
        name="mul",
    )
    _build_qdq_model(
        graph_name="mul_001_qdq_const",
        input_shape=[1, 8],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 8],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=const_nodes + [mul_node],
        runtime_initializers=const_inits,
        save_path=path,
    )
    return path


def gen_sub_001() -> Path:
    """SUB_001: official Sub QDQ with constant second input — [1,8] -> [1,8]."""
    path = MODELS_ROOT / "core" / "sub" / "SUB_001.onnx"
    const_nodes, const_inits, _const_vis, const_dq = _make_qdq_constant(
        prefix="sub.const",
        values=np.array([1, -2, 2, -1, 3, -3, 1, 0], dtype=np.int8).reshape(1, 8),
        scale=0.05,
        zero_point=0,
    )
    sub_node = helper.make_node("Sub", ["input_dq", const_dq], ["output"], name="sub")
    _build_qdq_model(
        graph_name="sub_001_qdq_const",
        input_shape=[1, 8],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 8],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=const_nodes + [sub_node],
        runtime_initializers=const_inits,
        save_path=path,
    )
    return path


def gen_div_001() -> Path:
    """DIV_001: official Div QDQ with non-zero constant second input — [1,8] -> [1,8]."""
    path = MODELS_ROOT / "core" / "div" / "DIV_001.onnx"
    const_nodes, const_inits, _const_vis, const_dq = _make_qdq_constant(
        prefix="div.const",
        values=np.array([2, 1, -2, -1, 4, -4, 2, -2], dtype=np.int8).reshape(1, 8),
        scale=0.05,
        zero_point=0,
    )
    div_node = helper.make_node("Div", ["input_dq", const_dq], ["output"], name="div")
    _build_qdq_model(
        graph_name="div_001_qdq_const",
        input_shape=[1, 8],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 8],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=const_nodes + [div_node],
        runtime_initializers=const_inits,
        save_path=path,
    )
    return path


def gen_sigmoid_001() -> Path:
    """SIGMOID_001: official Sigmoid QDQ — [1,8] -> [1,8]."""
    path = MODELS_ROOT / "core" / "sigmoid" / "SIGMOID_001.onnx"
    sigmoid_node = helper.make_node("Sigmoid", ["input_dq"], ["output"], name="sigmoid")
    _build_qdq_model(
        graph_name="sigmoid_001_qdq",
        input_shape=[1, 8],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 8],
        output_scale=1.0 / 256.0,
        output_zp=-128,
        runtime_nodes=[sigmoid_node],
        runtime_initializers=[],
        save_path=path,
    )
    return path


def gen_tanh_001() -> Path:
    """TANH_001: official Tanh QDQ — [1,8] -> [1,8]."""
    path = MODELS_ROOT / "core" / "tanh" / "TANH_001.onnx"
    tanh_node = helper.make_node("Tanh", ["input_dq"], ["output"], name="tanh")
    _build_qdq_model(
        graph_name="tanh_001_qdq",
        input_shape=[1, 8],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 8],
        output_scale=1.0 / 128.0,
        output_zp=0,
        runtime_nodes=[tanh_node],
        runtime_initializers=[],
        save_path=path,
    )
    return path


def gen_leakyrelu_001() -> Path:
    """LEAKYRELU_001: official LeakyRelu QDQ alpha=0.1 — [1,8] -> [1,8]."""
    path = MODELS_ROOT / "core" / "leakyrelu" / "LEAKYRELU_001.onnx"
    node = helper.make_node(
        "LeakyRelu",
        ["input_dq"],
        ["output"],
        name="leakyrelu",
        alpha=0.1,
    )
    _build_qdq_model(
        graph_name="leakyrelu_001_qdq",
        input_shape=[1, 8],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 8],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[node],
        runtime_initializers=[],
        save_path=path,
    )
    return path


def gen_clip_001() -> Path:
    """CLIP_001: official Clip QDQ with constant min/max — [1,8] -> [1,8]."""
    path = MODELS_ROOT / "core" / "clip" / "CLIP_001.onnx"
    min_name = "clip.min"
    max_name = "clip.max"
    node = helper.make_node(
        "Clip",
        ["input_dq", min_name, max_name],
        ["output"],
        name="clip",
    )
    _build_qdq_model(
        graph_name="clip_001_qdq",
        input_shape=[1, 8],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 8],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[node],
        runtime_initializers=[
            numpy_helper.from_array(np.array(-0.10, dtype=np.float32), name=min_name),
            numpy_helper.from_array(np.array(0.20, dtype=np.float32), name=max_name),
        ],
        save_path=path,
    )
    return path


def gen_negop_001() -> Path:
    """NEGOP_001: official Neg QDQ — [1,8] -> [1,8]."""
    path = MODELS_ROOT / "core" / "neg" / "NEGOP_001.onnx"
    node = helper.make_node("Neg", ["input_dq"], ["output"], name="neg")
    _build_qdq_model(
        graph_name="negop_001_qdq",
        input_shape=[1, 8],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 8],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[node],
        runtime_initializers=[],
        save_path=path,
    )
    return path


def gen_sqrt_001() -> Path:
    """SQRT_001: official Sqrt QDQ with non-negative input domain — [1,8] -> [1,8]."""
    path = MODELS_ROOT / "core" / "sqrt" / "SQRT_001.onnx"
    node = helper.make_node("Sqrt", ["input_dq"], ["output"], name="sqrt")
    _build_qdq_model(
        graph_name="sqrt_001_qdq",
        input_shape=[1, 8],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 8],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[node],
        runtime_initializers=[],
        save_path=path,
    )
    return path


def gen_reciprocal_001() -> Path:
    """RECIPROCAL_001: official Reciprocal QDQ with finite non-zero input — [1,8] -> [1,8]."""
    path = MODELS_ROOT / "core" / "reciprocal" / "RECIPROCAL_001.onnx"
    node = helper.make_node("Reciprocal", ["input_dq"], ["output"], name="reciprocal")
    _build_qdq_model(
        graph_name="reciprocal_001_qdq",
        input_shape=[1, 8],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 8],
        output_scale=0.02,
        output_zp=0,
        runtime_nodes=[node],
        runtime_initializers=[],
        save_path=path,
    )
    return path


def gen_reducemean_001() -> Path:
    """REDUCEMEAN_001: official ReduceMean QDQ axis=1 keepdims=1 — [2,4] -> [2,1]."""
    path = MODELS_ROOT / "core" / "reducemean" / "REDUCEMEAN_001.onnx"
    node = helper.make_node(
        "ReduceMean",
        ["input_dq"],
        ["output"],
        name="reducemean",
        axes=[1],
        keepdims=1,
    )
    _build_qdq_model(
        graph_name="reducemean_001_qdq",
        input_shape=[2, 4],
        input_scale=0.05,
        input_zp=0,
        output_shape=[2, 1],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[node],
        runtime_initializers=[],
        save_path=path,
    )
    return path


def gen_unsqueeze_001() -> Path:
    """UNSQUEEZE_001: official Unsqueeze QDQ data path — [1,4] -> [1,1,4]."""
    path = MODELS_ROOT / "core" / "unsqueeze" / "UNSQUEEZE_001.onnx"
    axes_name = "unsqueeze.axes"
    node = helper.make_node(
        "Unsqueeze",
        ["input_dq", axes_name],
        ["output"],
        name="unsqueeze",
    )
    _build_qdq_model(
        graph_name="unsqueeze_001_qdq",
        input_shape=[1, 4],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 1, 4],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[node],
        runtime_initializers=[
            numpy_helper.from_array(np.array([1], dtype=np.int64), name=axes_name)
        ],
        save_path=path,
    )
    return path


def gen_exp_001() -> Path:
    """EXP_001: official Exp QDQ with bounded input — [1,8] -> [1,8]."""
    path = MODELS_ROOT / "core" / "exp" / "EXP_001.onnx"
    node = helper.make_node("Exp", ["input_dq"], ["output"], name="exp")
    _build_qdq_model(
        graph_name="exp_001_qdq",
        input_shape=[1, 8],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 8],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[node],
        runtime_initializers=[],
        save_path=path,
    )
    return path


def gen_log_001() -> Path:
    """LOG_001: official Log QDQ with positive input domain — [1,8] -> [1,8]."""
    path = MODELS_ROOT / "core" / "log" / "LOG_001.onnx"
    node = helper.make_node("Log", ["input_dq"], ["output"], name="log")
    _build_qdq_model(
        graph_name="log_001_qdq",
        input_shape=[1, 8],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 8],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[node],
        runtime_initializers=[],
        save_path=path,
    )
    return path


def gen_floor_001() -> Path:
    """FLOOR_001: official Floor QDQ — [1,8] -> [1,8]."""
    path = MODELS_ROOT / "core" / "floor" / "FLOOR_001.onnx"
    node = helper.make_node("Floor", ["input_dq"], ["output"], name="floor")
    _build_qdq_model(
        graph_name="floor_001_qdq",
        input_shape=[1, 8],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 8],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[node],
        runtime_initializers=[],
        save_path=path,
    )
    return path


def gen_ceil_001() -> Path:
    """CEIL_001: official Ceil QDQ — [1,8] -> [1,8]."""
    path = MODELS_ROOT / "core" / "ceil" / "CEIL_001.onnx"
    node = helper.make_node("Ceil", ["input_dq"], ["output"], name="ceil")
    _build_qdq_model(
        graph_name="ceil_001_qdq",
        input_shape=[1, 8],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 8],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[node],
        runtime_initializers=[],
        save_path=path,
    )
    return path


def gen_round_001() -> Path:
    """ROUND_001: official Round QDQ — [1,8] -> [1,8]."""
    path = MODELS_ROOT / "core" / "round" / "ROUND_001.onnx"
    node = helper.make_node("Round", ["input_dq"], ["output"], name="round")
    _build_qdq_model(
        graph_name="round_001_qdq",
        input_shape=[1, 8],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 8],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[node],
        runtime_initializers=[],
        save_path=path,
    )
    return path


def gen_sign_001() -> Path:
    """SIGN_001: official Sign QDQ — [1,8] -> [1,8]."""
    path = MODELS_ROOT / "core" / "sign" / "SIGN_001.onnx"
    node = helper.make_node("Sign", ["input_dq"], ["output"], name="sign")
    _build_qdq_model(
        graph_name="sign_001_qdq",
        input_shape=[1, 8],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 8],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[node],
        runtime_initializers=[],
        save_path=path,
    )
    return path


def gen_min_001() -> Path:
    """MIN_001: official Min QDQ with constant second input — [1,8] -> [1,8]."""
    path = MODELS_ROOT / "core" / "min" / "MIN_001.onnx"
    const_nodes, const_inits, _const_vis, const_dq = _make_qdq_constant(
        prefix="min.const",
        values=np.array([0, -4, 4, 0, 6, -2, 8, 4], dtype=np.int8).reshape(1, 8),
        scale=0.05,
        zero_point=0,
    )
    node = helper.make_node("Min", ["input_dq", const_dq], ["output"], name="min")
    _build_qdq_model(
        graph_name="min_001_qdq_const",
        input_shape=[1, 8],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 8],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=const_nodes + [node],
        runtime_initializers=const_inits,
        save_path=path,
    )
    return path


def gen_max_001() -> Path:
    """MAX_001: official Max QDQ with constant second input — [1,8] -> [1,8]."""
    path = MODELS_ROOT / "core" / "max" / "MAX_001.onnx"
    const_nodes, const_inits, _const_vis, const_dq = _make_qdq_constant(
        prefix="max.const",
        values=np.array([0, -4, 4, 0, 6, -2, 8, 4], dtype=np.int8).reshape(1, 8),
        scale=0.05,
        zero_point=0,
    )
    node = helper.make_node("Max", ["input_dq", const_dq], ["output"], name="max")
    _build_qdq_model(
        graph_name="max_001_qdq_const",
        input_shape=[1, 8],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 8],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=const_nodes + [node],
        runtime_initializers=const_inits,
        save_path=path,
    )
    return path


def gen_pow_001() -> Path:
    """POW_001: official Pow QDQ with constant exponent — [1,8] -> [1,8]."""
    path = MODELS_ROOT / "core" / "pow" / "POW_001.onnx"
    const_nodes, const_inits, _const_vis, const_dq = _make_qdq_constant(
        prefix="pow.const",
        values=np.array([10, 10, 10, 10, 10, 10, 10, 10], dtype=np.int8).reshape(1, 8),
        scale=0.05,
        zero_point=0,
    )
    node = helper.make_node("Pow", ["input_dq", const_dq], ["output"], name="pow")
    _build_qdq_model(
        graph_name="pow_001_qdq_const",
        input_shape=[1, 8],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 8],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=const_nodes + [node],
        runtime_initializers=const_inits,
        save_path=path,
    )
    return path


def gen_reducesum_001() -> Path:
    """REDUCESUM_001: official ReduceSum QDQ axis=1 keepdims=1 — [2,4] -> [2,1]."""
    path = MODELS_ROOT / "core" / "reducesum" / "REDUCESUM_001.onnx"
    axes_name = "reducesum.axes"
    node = helper.make_node(
        "ReduceSum",
        ["input_dq", axes_name],
        ["output"],
        name="reducesum",
        keepdims=1,
    )
    _build_qdq_model(
        graph_name="reducesum_001_qdq",
        input_shape=[2, 4],
        input_scale=0.05,
        input_zp=0,
        output_shape=[2, 1],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[node],
        runtime_initializers=[
            numpy_helper.from_array(np.array([1], dtype=np.int64), name=axes_name)
        ],
        save_path=path,
    )
    return path


def gen_reducemax_001() -> Path:
    """REDUCEMAX_001: official ReduceMax QDQ axis=1 keepdims=1 — [2,4] -> [2,1]."""
    path = MODELS_ROOT / "core" / "reducemax" / "REDUCEMAX_001.onnx"
    node = helper.make_node(
        "ReduceMax",
        ["input_dq"],
        ["output"],
        name="reducemax",
        axes=[1],
        keepdims=1,
    )
    _build_qdq_model(
        graph_name="reducemax_001_qdq",
        input_shape=[2, 4],
        input_scale=0.05,
        input_zp=0,
        output_shape=[2, 1],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[node],
        runtime_initializers=[],
        save_path=path,
    )
    return path


def gen_reducemin_001() -> Path:
    """REDUCEMIN_001: official ReduceMin QDQ axis=1 keepdims=1 — [2,4] -> [2,1]."""
    path = MODELS_ROOT / "core" / "reducemin" / "REDUCEMIN_001.onnx"
    node = helper.make_node(
        "ReduceMin",
        ["input_dq"],
        ["output"],
        name="reducemin",
        axes=[1],
        keepdims=1,
    )
    _build_qdq_model(
        graph_name="reducemin_001_qdq",
        input_shape=[2, 4],
        input_scale=0.05,
        input_zp=0,
        output_shape=[2, 1],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[node],
        runtime_initializers=[],
        save_path=path,
    )
    return path


def gen_where_001() -> Path:
    """WHERE_001: official Where with static bool mask — [1,8] -> [1,8]."""
    path = MODELS_ROOT / "core" / "where" / "WHERE_001.onnx"
    cond_name = "where.cond"
    cond = np.array([True, False, True, False, False, True, True, False], dtype=np.bool_).reshape(1, 8)
    const_nodes, const_inits, _const_vis, const_dq = _make_qdq_constant(
        prefix="where.else",
        values=np.array([-6, -4, -2, 0, 2, 4, 6, 8], dtype=np.int8).reshape(1, 8),
        scale=0.05,
        zero_point=0,
    )
    node = helper.make_node("Where", [cond_name, "input_dq", const_dq], ["output"], name="where")
    _build_qdq_model(
        graph_name="where_001_qdq",
        input_shape=[1, 8],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 8],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=const_nodes + [node],
        runtime_initializers=[
            numpy_helper.from_array(cond, name=cond_name),
            *const_inits,
        ],
        save_path=path,
    )
    return path


def _gen_compare_where_case(
    *,
    case_id: str,
    op_type: str,
    category: str,
    const_values: np.ndarray,
) -> Path:
    path = MODELS_ROOT / "core" / category / f"{case_id}.onnx"
    const_nodes, const_inits, _const_vis, const_dq = _make_qdq_constant(
        prefix=f"{category}.rhs",
        values=const_values.astype(np.int8).reshape(1, 8),
        scale=0.05,
        zero_point=0,
    )
    zero_nodes, zero_inits, _zero_vis, zero_dq = _make_qdq_constant(
        prefix=f"{category}.zero",
        values=np.zeros((1, 8), dtype=np.int8),
        scale=0.05,
        zero_point=0,
    )
    compare_node = helper.make_node(op_type, ["input_dq", const_dq], [f"{category}.cond"], name=category)
    where_node = helper.make_node(
        "Where",
        [f"{category}.cond", "input_dq", zero_dq],
        ["output"],
        name=f"{category}.where",
    )
    _build_qdq_model(
        graph_name=f"{case_id.lower()}_qdq",
        input_shape=[1, 8],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 8],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=const_nodes + zero_nodes + [compare_node, where_node],
        runtime_initializers=const_inits + zero_inits,
        save_path=path,
    )
    return path


def gen_equal_001() -> Path:
    """EQUAL_001: Equal bool feeds Where — [1,8] -> [1,8]."""
    return _gen_compare_where_case(
        case_id="EQUAL_001",
        op_type="Equal",
        category="equal",
        const_values=np.array([0, -2, 2, 0, 3, -1, 4, 2], dtype=np.int8),
    )


def gen_greater_001() -> Path:
    """GREATER_001: Greater bool feeds Where — [1,8] -> [1,8]."""
    return _gen_compare_where_case(
        case_id="GREATER_001",
        op_type="Greater",
        category="greater",
        const_values=np.array([-2, -2, 0, 1, 2, 2, 4, 5], dtype=np.int8),
    )


def gen_less_001() -> Path:
    """LESS_001: Less bool feeds Where — [1,8] -> [1,8]."""
    return _gen_compare_where_case(
        case_id="LESS_001",
        op_type="Less",
        category="less",
        const_values=np.array([2, 1, 3, 1, 5, 0, 8, 6], dtype=np.int8),
    )


def gen_greaterorequal_001() -> Path:
    """GREATEROREQUAL_001: GreaterOrEqual bool feeds Where — [1,8] -> [1,8]."""
    return _gen_compare_where_case(
        case_id="GREATEROREQUAL_001",
        op_type="GreaterOrEqual",
        category="greaterorequal",
        const_values=np.array([0, 2, 2, 3, 5, 5, 6, 8], dtype=np.int8),
    )


def gen_lessorequal_001() -> Path:
    """LESSOREQUAL_001: LessOrEqual bool feeds Where — [1,8] -> [1,8]."""
    return _gen_compare_where_case(
        case_id="LESSOREQUAL_001",
        op_type="LessOrEqual",
        category="lessorequal",
        const_values=np.array([0, 2, 1, 3, 4, 6, 6, 7], dtype=np.int8),
    )


def _gen_bool_logic_case(
    *,
    case_id: str,
    category: str,
    op_a: str,
    rhs_a: np.ndarray,
    op_b: str | None,
    rhs_b: np.ndarray | None,
    logic_op: str,
) -> Path:
    """构造 compare -> bool-logic -> Where 的 QDQ/int8 用例。

    两个比较算子（或单个，Not 时 op_b=None）产 bool 条件，经逻辑算子和
    成 combined condition，再由 Where 做 int8 数据选择。bool 全部为内部
    中间张量，外部 ABI 仍为 int8 QDQ。
    """
    path = MODELS_ROOT / "core" / category / f"{case_id}.onnx"
    a_nodes, a_inits, _a_vis, a_dq = _make_qdq_constant(
        prefix=f"{category}.rhs_a",
        values=rhs_a.astype(np.int8).reshape(1, 8),
        scale=0.05,
        zero_point=0,
    )
    cond_a_name = f"{category}.a_cond"
    comp_a_node = helper.make_node(op_a, ["input_dq", a_dq], [cond_a_name], name=f"{category}.{op_a.lower()}_a")
    runtime_nodes = a_nodes + [comp_a_node]
    runtime_inits = list(a_inits)

    cond_b_name = None
    if op_b is not None and rhs_b is not None:
        b_nodes, b_inits, _b_vis, b_dq = _make_qdq_constant(
            prefix=f"{category}.rhs_b",
            values=rhs_b.astype(np.int8).reshape(1, 8),
            scale=0.05,
            zero_point=0,
        )
        cond_b_name = f"{category}.b_cond"
        comp_b_node = helper.make_node(op_b, ["input_dq", b_dq], [cond_b_name], name=f"{category}.{op_b.lower()}_b")
        runtime_nodes += b_nodes + [comp_b_node]
        runtime_inits += list(b_inits)

    combined_name = f"{category}.combined"
    logic_inputs = [cond_a_name] if cond_b_name is None else [cond_a_name, cond_b_name]
    logic_node = helper.make_node(logic_op, logic_inputs, [combined_name], name=f"{category}.{logic_op.lower()}")

    zero_nodes, zero_inits, _zero_vis, zero_dq = _make_qdq_constant(
        prefix=f"{category}.zero",
        values=np.zeros((1, 8), dtype=np.int8),
        scale=0.05,
        zero_point=0,
    )
    where_node = helper.make_node(
        "Where",
        [combined_name, "input_dq", zero_dq],
        ["output"],
        name=f"{category}.where",
    )
    _build_qdq_model(
        graph_name=f"{case_id.lower()}_qdq",
        input_shape=[1, 8],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 8],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=runtime_nodes + zero_nodes + [logic_node, where_node],
        runtime_initializers=runtime_inits + zero_inits,
        save_path=path,
    )
    return path


def gen_and_001() -> Path:
    """AND_001: Greater->And->Where 窗口条件 — [1,8] -> [1,8]."""
    return _gen_bool_logic_case(
        case_id="AND_001",
        category="and",
        op_a="Greater",
        rhs_a=np.array([2, 2, 2, 2, 2, 2, 2, 2], dtype=np.int8),
        op_b="Less",
        rhs_b=np.array([6, 6, 6, 6, 6, 6, 6, 6], dtype=np.int8),
        logic_op="And",
    )


def gen_or_001() -> Path:
    """OR_001: Less->Or->Where 双上界条件 — [1,8] -> [1,8]."""
    return _gen_bool_logic_case(
        case_id="OR_001",
        category="or",
        op_a="Less",
        rhs_a=np.array([2, 2, 2, 2, 2, 2, 2, 2], dtype=np.int8),
        op_b="Less",
        rhs_b=np.array([6, 6, 6, 6, 6, 6, 6, 6], dtype=np.int8),
        logic_op="Or",
    )


def gen_not_001() -> Path:
    """NOT_001: Greater->Not->Where 取反条件 — [1,8] -> [1,8]."""
    return _gen_bool_logic_case(
        case_id="NOT_001",
        category="not",
        op_a="Greater",
        rhs_a=np.array([2, 2, 2, 2, 2, 2, 2, 2], dtype=np.int8),
        op_b=None,
        rhs_b=None,
        logic_op="Not",
    )


def gen_xor_001() -> Path:
    """XOR_001: Greater->Xor->Where 窗口互斥条件 — [1,8] -> [1,8]."""
    return _gen_bool_logic_case(
        case_id="XOR_001",
        category="xor",
        op_a="Greater",
        rhs_a=np.array([2, 2, 2, 2, 2, 2, 2, 2], dtype=np.int8),
        op_b="Less",
        rhs_b=np.array([6, 6, 6, 6, 6, 6, 6, 6], dtype=np.int8),
        logic_op="Xor",
    )


def _gen_reduce_axis1_case(case_id: str, op_type: str, category: str, output_scale: float) -> Path:
    path = MODELS_ROOT / "core" / category / f"{case_id}.onnx"
    node = helper.make_node(
        op_type,
        ["input_dq"],
        ["output"],
        name=category,
        axes=[1],
        keepdims=1,
    )
    _build_qdq_model(
        graph_name=f"{case_id.lower()}_qdq",
        input_shape=[2, 4],
        input_scale=0.05,
        input_zp=0,
        output_shape=[2, 1],
        output_scale=output_scale,
        output_zp=0,
        runtime_nodes=[node],
        runtime_initializers=[],
        save_path=path,
    )
    return path


def gen_reduceprod_001() -> Path:
    """REDUCEPROD_001: official ReduceProd QDQ axis=1 keepdims=1 — [2,4] -> [2,1]."""
    return _gen_reduce_axis1_case("REDUCEPROD_001", "ReduceProd", "reduceprod", 0.01)


def gen_reducel1_001() -> Path:
    """REDUCEL1_001: official ReduceL1 QDQ axis=1 keepdims=1 — [2,4] -> [2,1]."""
    return _gen_reduce_axis1_case("REDUCEL1_001", "ReduceL1", "reducel1", 0.05)


def gen_reducel2_001() -> Path:
    """REDUCEL2_001: official ReduceL2 QDQ axis=1 keepdims=1 — [2,4] -> [2,1]."""
    return _gen_reduce_axis1_case("REDUCEL2_001", "ReduceL2", "reducel2", 0.05)


def gen_reducelogsum_001() -> Path:
    """REDUCELOGSUM_001: official ReduceLogSum QDQ axis=1 keepdims=1 — [2,4] -> [2,1]."""
    return _gen_reduce_axis1_case("REDUCELOGSUM_001", "ReduceLogSum", "reducelogsum", 0.05)


def gen_reducelogsumexp_001() -> Path:
    """REDUCELOGSUMEXP_001: official ReduceLogSumExp QDQ axis=1 keepdims=1 — [2,4] -> [2,1]."""
    return _gen_reduce_axis1_case("REDUCELOGSUMEXP_001", "ReduceLogSumExp", "reducelogsumexp", 0.05)


def gen_reducesumsquare_001() -> Path:
    """REDUCESUMSQUARE_001: official ReduceSumSquare QDQ axis=1 keepdims=1 — [2,4] -> [2,1]."""
    return _gen_reduce_axis1_case("REDUCESUMSQUARE_001", "ReduceSumSquare", "reducesumsquare", 0.05)


def _gen_spatial_pool_case(
    *,
    case_id: str,
    op_type: str,
    category: str,
    attrs: dict[str, object] | None = None,
) -> Path:
    """构造 rank=4 全局/窗口池化 QDQ 用例 — [1,2,2,2] 或 [1,1,2,2]。"""
    if case_id == "LPPOOL_001":
        in_shape, out_shape = [1, 1, 2, 2], [1, 1, 1, 1]
    else:
        in_shape, out_shape = [1, 2, 2, 2], [1, 2, 1, 1]
    path = MODELS_ROOT / "core" / category / f"{case_id}.onnx"
    node = helper.make_node(op_type, ["input_dq"], ["output"], name=category, **(attrs or {}))
    _build_qdq_model(
        graph_name=f"{case_id.lower()}_qdq",
        input_shape=in_shape,
        input_scale=0.05,
        input_zp=0,
        output_shape=out_shape,
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[node],
        runtime_initializers=[],
        save_path=path,
    )
    return path


def gen_globalmaxpool_001() -> Path:
    """GLOBALMAXPOOL_001: official GlobalMaxPool QDQ — [1,2,2,2] -> [1,2,1,1]."""
    return _gen_spatial_pool_case(case_id="GLOBALMAXPOOL_001", op_type="GlobalMaxPool", category="globalmaxpool")


def gen_globallppool_001() -> Path:
    """GLOBALLPPOOL_001: official GlobalLpPool QDQ — [1,2,2,2] -> [1,2,1,1]."""
    return _gen_spatial_pool_case(
        case_id="GLOBALLPPOOL_001", op_type="GlobalLpPool", category="globallppool", attrs={"p": 2}
    )


def gen_lppool_001() -> Path:
    """LPPOOL_001: official LpPool QDQ kernel=2 stride=1 — [1,1,2,2] -> [1,1,1,1]."""
    return _gen_spatial_pool_case(
        case_id="LPPOOL_001",
        op_type="LpPool",
        category="lppool",
        attrs={"p": 2, "kernel_shape": [2, 2], "strides": [1, 1]},
    )


def gen_lpnormalization_001() -> Path:
    """LPNORMALIZATION_001: official LpNormalization QDQ axis=1 p=2 — [2,4] -> [2,4]."""
    path = MODELS_ROOT / "core" / "lpnormalization" / "LPNORMALIZATION_001.onnx"
    node = helper.make_node("LpNormalization", ["input_dq"], ["output"], name="lpnormalization", axis=1, p=2)
    _build_qdq_model(
        graph_name="lpnormalization_001_qdq",
        input_shape=[2, 4],
        input_scale=0.05,
        input_zp=0,
        output_shape=[2, 4],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[node],
        runtime_initializers=[],
        save_path=path,
    )
    return path


def gen_cumsum_001() -> Path:
    """CUMSUM_001: official CumSum QDQ axis=1 — [2,4] -> [2,4]."""
    path = MODELS_ROOT / "core" / "cumsum" / "CUMSUM_001.onnx"
    axes_name = "cumsum.axis"
    node = helper.make_node("CumSum", ["input_dq", axes_name], ["output"], name="cumsum")
    _build_qdq_model(
        graph_name="cumsum_001_qdq",
        input_shape=[2, 4],
        input_scale=0.05,
        input_zp=0,
        output_shape=[2, 4],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[node],
        runtime_initializers=[numpy_helper.from_array(np.array([1], dtype=np.int64), name=axes_name)],
        save_path=path,
    )
    return path


def _gen_variadic_const_case(*, case_id: str, op_type: str, category: str, const_values: np.ndarray) -> Path:
    """构造双输入逐元素 QDQ 用例（第二输入为量化常量）— [1,8] -> [1,8]。"""
    path = MODELS_ROOT / "core" / category / f"{case_id}.onnx"
    const_nodes, const_inits, _vis, const_dq = _make_qdq_constant(
        prefix=f"{category}.rhs",
        values=const_values.astype(np.int8).reshape(1, 8),
        scale=0.05,
        zero_point=0,
    )
    node = helper.make_node(op_type, ["input_dq", const_dq], ["output"], name=category)
    _build_qdq_model(
        graph_name=f"{case_id.lower()}_qdq",
        input_shape=[1, 8],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 8],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=const_nodes + [node],
        runtime_initializers=const_inits,
        save_path=path,
    )
    return path


def gen_mean_001() -> Path:
    """MEAN_001: official Mean QDQ const rhs — [1,8] -> [1,8]."""
    return _gen_variadic_const_case(
        case_id="MEAN_001",
        op_type="Mean",
        category="mean",
        const_values=np.array([2, -2, 4, -4, 6, -6, 8, -8], dtype=np.int8),
    )


def gen_sum_001() -> Path:
    """SUM_001: official Sum QDQ const rhs — [1,8] -> [1,8]."""
    return _gen_variadic_const_case(
        case_id="SUM_001",
        op_type="Sum",
        category="sum",
        const_values=np.array([2, -2, 4, -4, 6, -6, 8, -8], dtype=np.int8),
    )


def gen_identity_001() -> Path:
    """IDENTITY_001: official Identity QDQ passthrough — [1,8] -> [1,8]."""
    return _gen_unary_simple_case(case_id="IDENTITY_001", op_type="Identity", category="identity", node_name="identity")


def gen_cast_001() -> Path:
    """CAST_001: official Cast int8-to-int8 between Q and DQ — [1,8] -> [1,8].

    Input(float) -> Q(int8) -> Cast(to=int8, identity) -> DQ(float) -> Output.
    Cast folds to a passthrough between the quantization boundaries.
    """
    path = MODELS_ROOT / "core" / "cast" / "CAST_001.onnx"
    shape = [1, 8]
    inp_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)
    out_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)
    q_name, q_scale, q_zp = "cast.q", "cast.q_scale", "cast.q_zp"
    inits = [
        numpy_helper.from_array(np.array(0.05, dtype=np.float32), name=q_scale),
        numpy_helper.from_array(np.array(0, dtype=np.int8), name=q_zp),
    ]
    q_node = helper.make_node("QuantizeLinear", ["input", q_scale, q_zp], ["cast_in"], name="cast_q")
    cast_node = helper.make_node("Cast", ["cast_in"], ["cast_out"], name="cast", to=TensorProto.INT8)
    dq_node = helper.make_node(
        "DequantizeLinear", ["cast_out", q_scale, q_zp], ["output"], name="cast_dq"
    )
    build_and_save(
        nodes=[q_node, cast_node, dq_node],
        inputs=[inp_vi],
        outputs=[out_vi],
        initializers=inits,
        save_path=path,
        graph_name="cast_001_qdq",
    )
    return path


def gen_pad_001() -> Path:
    """PAD_001: official Pad QDQ rank4 constant mode — [1,1,2,3] -> [1,1,4,5]."""
    path = MODELS_ROOT / "core" / "pad" / "PAD_001.onnx"
    pads_name = "pad.pads"
    pads = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
    pad_node = helper.make_node(
        "Pad",
        ["input_dq", pads_name],
        ["output"],
        name="pad",
        mode="constant",
    )
    _build_qdq_model(
        graph_name="pad_001_qdq",
        input_shape=[1, 1, 2, 3],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 1, 4, 5],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[pad_node],
        runtime_initializers=[numpy_helper.from_array(pads, name=pads_name)],
        save_path=path,
    )
    return path


def gen_slice_001() -> Path:
    """SLICE_001: official Slice QDQ static params — [1,1,3,4] -> [1,1,2,2]."""
    path = MODELS_ROOT / "core" / "slice" / "SLICE_001.onnx"
    params = {
        "slice.starts": np.array([0, 0, 1, 1], dtype=np.int64),
        "slice.ends": np.array([1, 1, 3, 3], dtype=np.int64),
        "slice.axes": np.array([0, 1, 2, 3], dtype=np.int64),
        "slice.steps": np.array([1, 1, 1, 1], dtype=np.int64),
    }
    slice_node = helper.make_node(
        "Slice",
        ["input_dq", "slice.starts", "slice.ends", "slice.axes", "slice.steps"],
        ["output"],
        name="slice",
    )
    _build_qdq_model(
        graph_name="slice_001_qdq",
        input_shape=[1, 1, 3, 4],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 1, 2, 2],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[slice_node],
        runtime_initializers=[
            numpy_helper.from_array(value, name=name) for name, value in params.items()
        ],
        save_path=path,
    )
    return path


def gen_gather_001() -> Path:
    """GATHER_001: official Gather QDQ static indices — [1,4] axis=1 -> [1,2]."""
    path = MODELS_ROOT / "core" / "gather" / "GATHER_001.onnx"
    indices_name = "gather.indices"
    gather_node = helper.make_node(
        "Gather",
        ["input_dq", indices_name],
        ["output"],
        name="gather",
        axis=1,
    )
    _build_qdq_model(
        graph_name="gather_001_qdq",
        input_shape=[1, 4],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 2],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[gather_node],
        runtime_initializers=[
            numpy_helper.from_array(np.array([2, 0], dtype=np.int64), name=indices_name)
        ],
        save_path=path,
    )
    return path


def gen_transpose_001() -> Path:
    """TRANSPOSE_001: official Transpose QDQ — [1,2,2,3] -> [1,2,3,2]."""
    path = MODELS_ROOT / "core" / "transpose" / "TRANSPOSE_001.onnx"
    transpose_node = helper.make_node(
        "Transpose",
        ["input_dq"],
        ["output"],
        name="transpose",
        perm=[0, 2, 3, 1],
    )
    _build_qdq_model(
        graph_name="transpose_001_qdq",
        input_shape=[1, 2, 2, 3],
        input_scale=0.05,
        input_zp=0,
        output_shape=[1, 2, 3, 2],
        output_scale=0.05,
        output_zp=0,
        runtime_nodes=[transpose_node],
        runtime_initializers=[],
        save_path=path,
    )
    return path


# ---------------------------------------------------------------------------
# TOPOLOGY 用例生成
# ---------------------------------------------------------------------------

def gen_topo_001() -> Path:
    """TOPO_001: Conv→Relu→Pool→Flatten→FC — [1,3,8,8] → [1,10]"""
    path = MODELS_ROOT / "topology" / "TOPO_001.onnx"

    nodes = []
    initializers = []
    value_infos = []

    input_name = "input"
    input_shape = [1, 3, 8, 8]
    output_shape = [1, 10]

    inp_vi = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
    out_vi = helper.make_tensor_value_info("fc_out_dq", TensorProto.FLOAT, output_shape)

    in_nodes, in_inits, in_vis = make_qdq_wrapper(
        "input_qdq", input_name, input_shape, 0.01, 0, is_input=True
    )
    nodes.extend(in_nodes)
    initializers.extend(in_inits)
    value_infos.extend(in_vis)

    conv_nodes, conv_inits = make_conv_node(
        "conv", ["input_dq"], ["conv_out"],
        weight_shape=[8, 3, 3, 3], kernel_shape=[3, 3],
        strides=[1, 1], pads=[1, 1, 1, 1], with_bias=True,
    )
    nodes.extend(conv_nodes)
    initializers.extend(conv_inits)

    relu_node = make_relu_node("relu", ["conv_out"], ["relu_out"])
    nodes.append(relu_node)

    mid_nodes, mid_inits, mid_vis = make_qdq_wrapper(
        "mid_qdq", "relu_out", [1, 8, 8, 8], 0.03, 0, is_input=False
    )
    nodes.extend(mid_nodes)
    initializers.extend(mid_inits)
    value_infos.extend(mid_vis)

    pool_in_nodes, pool_in_inits, pool_in_vis = make_qdq_wrapper(
        "pool_in_qdq", "relu_out_dq", [1, 8, 8, 8], 0.03, 0, is_input=True
    )
    nodes.extend(pool_in_nodes)
    initializers.extend(pool_in_inits)
    value_infos.extend(pool_in_vis)

    pool_node = make_maxpool_node(
        "maxpool", ["relu_out_dq_dq"], ["pool_out"],
        kernel_shape=[2, 2], strides=[2, 2],
    )
    nodes.append(pool_node)

    pool_out_nodes, pool_out_inits, pool_out_vis = make_qdq_wrapper(
        "pool_out_qdq", "pool_out", [1, 8, 4, 4], 0.03, 0, is_input=False
    )
    nodes.extend(pool_out_nodes)
    initializers.extend(pool_out_inits)
    value_infos.extend(pool_out_vis)

    flatten_in_nodes, flatten_in_inits, flatten_in_vis = make_qdq_wrapper(
        "flatten_in_qdq", "pool_out_dq", [1, 8, 4, 4], 0.03, 0, is_input=True
    )
    nodes.extend(flatten_in_nodes)
    initializers.extend(flatten_in_inits)
    value_infos.extend(flatten_in_vis)

    flatten_node = make_flatten_node("flatten", ["pool_out_dq_dq"], ["flat_out"], axis=1)
    nodes.append(flatten_node)

    fc_in_nodes, fc_in_inits, fc_in_vis = make_qdq_wrapper(
        "fc_in_qdq", "flat_out", [1, 128], 0.03, 0, is_input=False
    )
    nodes.extend(fc_in_nodes)
    initializers.extend(fc_in_inits)
    value_infos.extend(fc_in_vis)

    fc_dq_nodes, fc_dq_inits, fc_dq_vis = make_qdq_wrapper(
        "fc_dq_qdq", "flat_out_dq", [1, 128], 0.03, 0, is_input=True
    )
    nodes.extend(fc_dq_nodes)
    initializers.extend(fc_dq_inits)
    value_infos.extend(fc_dq_vis)

    gemm_nodes, gemm_inits = make_gemm_node(
        "fc", ["flat_out_dq_dq"], ["fc_out"], weight_shape=[10, 128], with_bias=True
    )
    nodes.extend(gemm_nodes)
    initializers.extend(gemm_inits)

    out_nodes, out_inits, out_vis = make_qdq_wrapper(
        "output_qdq", "fc_out", output_shape, 0.05, 0, is_input=False
    )
    nodes.extend(out_nodes)
    initializers.extend(out_inits)
    value_infos.extend(out_vis)

    build_and_save(
        nodes=nodes,
        inputs=[inp_vi],
        outputs=[out_vi],
        initializers=initializers,
        save_path=path,
        graph_name="topo_001",
        value_infos=value_infos,
    )
    return path


def gen_topo_002() -> Path:
    """TOPO_002: Conv1d→Relu→Conv1d→Relu→Flatten→FC — [1,1,10] → [1,3]"""
    path = MODELS_ROOT / "topology" / "TOPO_002.onnx"

    nodes = []
    initializers = []
    value_infos = []

    input_name = "input"
    input_shape = [1, 1, 10]
    output_shape = [1, 3]

    inp_vi = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
    out_vi = helper.make_tensor_value_info("fc_out_dq", TensorProto.FLOAT, output_shape)

    in_nodes, in_inits, in_vis = make_qdq_wrapper(
        "input_qdq", input_name, input_shape, 0.02, 0, is_input=True
    )
    nodes.extend(in_nodes)
    initializers.extend(in_inits)
    value_infos.extend(in_vis)

    conv1_nodes, conv1_inits = make_conv_node(
        "conv1", ["input_dq"], ["conv1_out"],
        weight_shape=[8, 1, 2], kernel_shape=[2],
        strides=[1], pads=[0, 0], with_bias=True,
    )
    nodes.extend(conv1_nodes)
    initializers.extend(conv1_inits)
    value_infos.append(helper.make_tensor_value_info("conv1_out", TensorProto.FLOAT, [1, 8, 9]))

    nodes.append(make_relu_node("relu1", ["conv1_out"], ["relu1_out"]))
    value_infos.append(helper.make_tensor_value_info("relu1_out", TensorProto.FLOAT, [1, 8, 9]))

    relu1_nodes, relu1_inits, relu1_vis = make_qdq_wrapper(
        "relu1_qdq", "relu1_out", [1, 8, 9], 0.03, 0, is_input=False
    )
    nodes.extend(relu1_nodes)
    initializers.extend(relu1_inits)
    value_infos.extend(relu1_vis)

    conv2_in_nodes, conv2_in_inits, conv2_in_vis = make_qdq_wrapper(
        "conv2_in_qdq", "relu1_out_dq", [1, 8, 9], 0.03, 0, is_input=True
    )
    nodes.extend(conv2_in_nodes)
    initializers.extend(conv2_in_inits)
    value_infos.extend(conv2_in_vis)
    value_infos.append(
        helper.make_tensor_value_info("relu1_out_dq_dq", TensorProto.FLOAT, [1, 8, 9])
    )

    conv2_nodes, conv2_inits = make_conv_node(
        "conv2", ["relu1_out_dq_dq"], ["conv2_out"],
        weight_shape=[8, 8, 2], kernel_shape=[2],
        strides=[1], pads=[0, 0], with_bias=True,
    )
    nodes.extend(conv2_nodes)
    initializers.extend(conv2_inits)
    value_infos.append(helper.make_tensor_value_info("conv2_out", TensorProto.FLOAT, [1, 8, 8]))

    nodes.append(make_relu_node("relu2", ["conv2_out"], ["relu2_out"]))
    value_infos.append(helper.make_tensor_value_info("relu2_out", TensorProto.FLOAT, [1, 8, 8]))

    relu2_nodes, relu2_inits, relu2_vis = make_qdq_wrapper(
        "relu2_qdq", "relu2_out", [1, 8, 8], 0.03, 0, is_input=False
    )
    nodes.extend(relu2_nodes)
    initializers.extend(relu2_inits)
    value_infos.extend(relu2_vis)

    flatten_in_nodes, flatten_in_inits, flatten_in_vis = make_qdq_wrapper(
        "flatten_in_qdq", "relu2_out_dq", [1, 8, 8], 0.03, 0, is_input=True
    )
    nodes.extend(flatten_in_nodes)
    initializers.extend(flatten_in_inits)
    value_infos.extend(flatten_in_vis)
    value_infos.append(
        helper.make_tensor_value_info("relu2_out_dq_dq", TensorProto.FLOAT, [1, 8, 8])
    )

    nodes.append(make_flatten_node("flatten", ["relu2_out_dq_dq"], ["flat_out"], axis=1))

    fc_in_nodes, fc_in_inits, fc_in_vis = make_qdq_wrapper(
        "fc_in_qdq", "flat_out", [1, 64], 0.03, 0, is_input=False
    )
    nodes.extend(fc_in_nodes)
    initializers.extend(fc_in_inits)
    value_infos.extend(fc_in_vis)

    fc_dq_nodes, fc_dq_inits, fc_dq_vis = make_qdq_wrapper(
        "fc_dq_qdq", "flat_out_dq", [1, 64], 0.03, 0, is_input=True
    )
    nodes.extend(fc_dq_nodes)
    initializers.extend(fc_dq_inits)
    value_infos.extend(fc_dq_vis)

    gemm_nodes, gemm_inits = make_gemm_node(
        "fc", ["flat_out_dq_dq"], ["fc_out"], weight_shape=[3, 64], with_bias=True
    )
    nodes.extend(gemm_nodes)
    initializers.extend(gemm_inits)

    out_nodes, out_inits, out_vis = make_qdq_wrapper(
        "output_qdq", "fc_out", output_shape, 0.05, 0, is_input=False
    )
    nodes.extend(out_nodes)
    initializers.extend(out_inits)
    value_infos.extend(out_vis)

    build_and_save(
        nodes=nodes,
        inputs=[inp_vi],
        outputs=[out_vi],
        initializers=initializers,
        save_path=path,
        graph_name="topo_002_conv1d_signal_jump",
        value_infos=value_infos,
    )
    return path


def gen_topo_003() -> Path:
    """TOPO_003: 真实 MNIST QLinear int8 模型。"""
    path = MODELS_ROOT / "topology" / "TOPO_003.onnx"
    source = FIXTURES_ROOT / "onnx" / "mnist-12-int8.onnx"
    if not source.exists():
        raise FileNotFoundError(f"missing fixture: {source}")
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, path)
    return path


# ---------------------------------------------------------------------------
# NETWORK 用例生成
# ---------------------------------------------------------------------------

def gen_net_001() -> Path:
    """NET_001: 真实 SqueezeNet 1.0 int8 ONNX 导入评估。"""
    return _copy_network_fixture("NET_001", "squeezenet1.0-12-int8.onnx")


def gen_net_002() -> Path:
    """NET_002: 真实 MobileNetV2 int8/QLinear 图像分类网络导入评估。"""
    return _copy_network_fixture("NET_002", "mobilenetv2-12-int8.onnx")


def gen_net_003() -> Path:
    """NET_003: 真实 SSD-MobileNet int8 目标检测网络导入评估。"""
    return _copy_network_fixture("NET_003", "ssd_mobilenet_v1_12-int8.onnx")


def gen_net_004() -> Path:
    """NET_004: KWS DS-CNN-style int8 网络导入评估。"""
    return _copy_network_fixture("NET_004", "keyword_spotting_dscnn.int8.onnx")


def gen_net_005() -> Path:
    """NET_005: Tiny signal jump int8 时序分类网络导入评估。"""
    return _copy_network_fixture("NET_005", "signal_jump.int8.onnx")


def gen_net_006() -> Path:
    """NET_006: EfficientNet-Lite4 int8 小型图像分类网络导入评估。"""
    return _copy_network_fixture("NET_006", "efficientnet-lite4-11-int8.onnx")


def _copy_network_fixture(case_id: str, fixture_name: str) -> Path:
    path = MODELS_ROOT / "networks" / f"{case_id}.onnx"
    source = FIXTURES_ROOT / "onnx" / fixture_name
    if not source.exists():
        raise FileNotFoundError(f"missing fixture: {source}")
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, path)
    return path


# ---------------------------------------------------------------------------
# NEGATIVE 用例生成
# ---------------------------------------------------------------------------

def gen_neg_001() -> Path:
    """NEG_001: float32 无 Q/DQ 模型"""
    path = MODELS_ROOT / "negative" / "NEG_001.onnx"

    input_name = "input"
    output_name = "output"
    input_shape = [1, 3, 8, 8]
    output_shape = [1, 8, 8, 8]

    inp_vi = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, input_shape)
    out_vi = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, output_shape)

    rng = np.random.RandomState(42)
    w_name = "conv.weight"
    b_name = "conv.bias"
    weight_arr = (rng.randn(8, 3, 3, 3) * 0.1).astype(np.float32)
    bias_arr = (rng.randn(8) * 0.01).astype(np.float32)
    w_init = numpy_helper.from_array(weight_arr, name=w_name)
    b_init = numpy_helper.from_array(bias_arr, name=b_name)

    conv_node = helper.make_node(
        "Conv", [input_name, w_name, b_name], ["conv_out"],
        name="conv", kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1],
    )
    relu_node = helper.make_node("Relu", ["conv_out"], [output_name], name="relu")

    build_and_save(
        nodes=[conv_node, relu_node],
        inputs=[inp_vi],
        outputs=[out_vi],
        initializers=[w_init, b_init],
        save_path=path,
        graph_name="neg_001_float32",
    )
    return path


def _gen_arg_neg_case(*, case_id: str, op_type: str, node_name: str, graph_name: str) -> Path:
    """ArgMax/ArgMin 非 int8 输出边界模型 — [1,4] int8 -> int64[1] index。"""
    path = MODELS_ROOT / "negative" / f"{case_id}.onnx"
    in_shape = [1, 4]
    inp_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, in_shape)
    out_vi = helper.make_tensor_value_info("output", TensorProto.INT64, [1])
    nodes, inits, vis = make_qdq_wrapper(
        "input_qdq", "input", in_shape, 0.05, 0, is_input=True
    )
    arg_node = helper.make_node(
        op_type, ["input_dq"], ["output"], name=node_name, axis=1, keepdims=1
    )
    build_and_save(
        nodes=nodes + [arg_node],
        inputs=[inp_vi],
        outputs=[out_vi],
        initializers=inits,
        value_infos=vis,
        save_path=path,
        graph_name=graph_name,
    )
    return path


def gen_neg_002() -> Path:
    """NEG_002: ArgMax int64 输出边界 — unsupported。"""
    return _gen_arg_neg_case(case_id="NEG_002", op_type="ArgMax", node_name="argmax", graph_name="neg_002_argmax")


def gen_neg_003() -> Path:
    """NEG_003: ArgMin int64 输出边界 — unsupported。"""
    return _gen_arg_neg_case(case_id="NEG_003", op_type="ArgMin", node_name="argmin", graph_name="neg_003_argmin")


# ---------------------------------------------------------------------------
# QLINEAR 用例生成
# ---------------------------------------------------------------------------

def gen_qlinear_num_001() -> Path:
    """QLINEAR_NUM_001: 最小 QLinearConv uint8 输入数值精度。

    模型结构:
        float[1,1,2,2] → Q(scale=1,zp=0,uint8) → QLinearConv(1×1,1ch) → DQ(scale=0.01,zp=112)
    权重: [[[[42]]]] (int8), w_scale=0.02, w_zp=0
    """
    path = MODELS_ROOT / "core" / "qlinear" / "QLINEAR_NUM_001.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)

    input_name = "input"
    output_name = "output"

    x_scale_init = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="x_scale")
    x_zp_init = numpy_helper.from_array(np.array([0], dtype=np.uint8), name="x_zp")
    w_scale_init = numpy_helper.from_array(np.array([0.02], dtype=np.float32), name="w_scale")
    w_zp_init = numpy_helper.from_array(np.array([0], dtype=np.uint8), name="w_zp")
    y_scale_init = numpy_helper.from_array(np.array([0.01], dtype=np.float32), name="y_scale")
    y_zp_init = numpy_helper.from_array(np.array([112], dtype=np.uint8), name="y_zp")

    weight_arr = np.array([42], dtype=np.int8).reshape(1, 1, 1, 1)
    weight_init = numpy_helper.from_array(weight_arr, name="weight")

    initializers = [
        x_scale_init, x_zp_init,
        w_scale_init, w_zp_init,
        y_scale_init, y_zp_init,
        weight_init,
    ]

    nodes = [
        helper.make_node(
            "QuantizeLinear",
            [input_name, "x_scale", "x_zp"],
            ["input_q"],
            name="input_quant",
        ),
        helper.make_node(
            "QLinearConv",
            ["input_q", "x_scale", "x_zp", "weight", "w_scale", "w_zp", "y_scale", "y_zp"],
            ["conv_out"],
            name="conv",
            kernel_shape=[1, 1],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
            group=1,
        ),
        helper.make_node(
            "DequantizeLinear",
            ["conv_out", "y_scale", "y_zp"],
            [output_name],
            name="output_dequant",
        ),
    ]

    inp_vi = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, 1, 2, 2])
    out_vi = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 1, 2, 2])

    build_and_save(
        nodes=nodes,
        inputs=[inp_vi],
        outputs=[out_vi],
        initializers=initializers,
        save_path=path,
        graph_name="qlinear_num_001",
    )
    return path


def gen_qlinear_num_002() -> Path:
    """QLINEAR_NUM_002: 多通道 QLinearConv + bias + per-channel scale。

    模型结构:
        float[1,2,2,2] → Q(scale=0.01,zp=128) → QLinearConv(2ch→2ch, bias, per-ch scale) → DQ(scale=0.005,zp=128)
    """
    path = MODELS_ROOT / "core" / "qlinear" / "QLINEAR_NUM_002.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)

    input_name = "input"
    output_name = "output"

    x_scale_init = numpy_helper.from_array(np.array([0.01], dtype=np.float32), name="x_scale")
    x_zp_init = numpy_helper.from_array(np.array([128], dtype=np.uint8), name="x_zp")
    w_scale_init = numpy_helper.from_array(np.array([0.02, 0.03], dtype=np.float32), name="w_scale")
    w_zp_init = numpy_helper.from_array(np.array([0, 0], dtype=np.int8), name="w_zp")
    y_scale_init = numpy_helper.from_array(np.array([0.005], dtype=np.float32), name="y_scale")
    y_zp_init = numpy_helper.from_array(np.array([128], dtype=np.uint8), name="y_zp")

    weight_arr = np.array([2, 1, 3, -1], dtype=np.int8).reshape(2, 2, 1, 1)
    weight_init = numpy_helper.from_array(weight_arr, name="weight")
    bias_arr = np.array([5, -5], dtype=np.int32)
    bias_init = numpy_helper.from_array(bias_arr, name="bias")

    initializers = [
        x_scale_init, x_zp_init, w_scale_init, w_zp_init,
        y_scale_init, y_zp_init, weight_init, bias_init,
    ]

    nodes = [
        helper.make_node("QuantizeLinear", [input_name, "x_scale", "x_zp"], ["input_q"], name="input_quant"),
        helper.make_node("QLinearConv",
            ["input_q", "x_scale", "x_zp", "weight", "w_scale", "w_zp", "y_scale", "y_zp", "bias"],
            ["conv_out"], name="conv",
            kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], group=1),
        helper.make_node("DequantizeLinear", ["conv_out", "y_scale", "y_zp"], [output_name], name="output_dequant"),
    ]

    inp_vi = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, 2, 2, 2])
    out_vi = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 2, 2, 2])

    build_and_save(
        nodes=nodes, inputs=[inp_vi], outputs=[out_vi],
        initializers=initializers, save_path=path, graph_name="qlinear_num_002",
    )
    return path
def gen_qlinear_num_003() -> Path:
    """QLINEAR_NUM_003: 高通道 5×5 QLinearConv (仿 MNIST Conv1)。"""
    path = MODELS_ROOT / "core" / "qlinear" / "QLINEAR_NUM_003.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)
    out_ch, in_ch, kh, kw = 8, 1, 5, 5
    input_name, output_name = "input", "output"
    x_scale_init = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="x_scale")
    x_zp_init = numpy_helper.from_array(np.array([0], dtype=np.uint8), name="x_zp")
    w_scales = np.array([0.008, 0.0045, 0.0077, 0.0038, 0.0054, 0.0058, 0.0044, 0.0045], dtype=np.float32)
    w_scale_init = numpy_helper.from_array(w_scales, name="w_scale")
    w_zp_init = numpy_helper.from_array(np.zeros(8, dtype=np.int8), name="w_zp")
    y_scale_init = numpy_helper.from_array(np.array([3.68], dtype=np.float32), name="y_scale")
    y_zp_init = numpy_helper.from_array(np.array([0], dtype=np.uint8), name="y_zp")
    weight_arr = (rng.randn(out_ch, in_ch, kh, kw) * 20).clip(-128, 127).astype(np.int8)
    weight_init = numpy_helper.from_array(weight_arr, name="weight")
    bias_arr = (rng.randn(out_ch) * 50).astype(np.int32)
    bias_init = numpy_helper.from_array(bias_arr, name="bias")
    initializers = [x_scale_init, x_zp_init, w_scale_init, w_zp_init,
                    y_scale_init, y_zp_init, weight_init, bias_init]
    nodes = [
        helper.make_node("QuantizeLinear", [input_name, "x_scale", "x_zp"], ["input_q"], name="iq"),
        helper.make_node("QLinearConv",
            ["input_q", "x_scale", "x_zp", "weight", "w_scale", "w_zp", "y_scale", "y_zp", "bias"],
            ["conv_out"], name="conv", kernel_shape=[5,5], strides=[1,1], pads=[0,0,0,0], group=1),
        helper.make_node("DequantizeLinear", ["conv_out", "y_scale", "y_zp"], [output_name], name="odq"),
    ]
    inp_vi = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1, 1, 8, 8])
    out_vi = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 8, 4, 4])
    build_and_save(nodes=nodes, inputs=[inp_vi], outputs=[out_vi],
                   initializers=initializers, save_path=path, graph_name="qlinear_num_003")
    return path



# ---------------------------------------------------------------------------
# 生成函数映射（必须与 cases_registry.CASE_MAP 中的 case_id 保持一一对应）
# ---------------------------------------------------------------------------

_GENERATORS: dict[str, object] = {
    "ABS_001": gen_abs_001,
    "GEMM_001": gen_gemm_001,
    "GEMM_002": gen_gemm_002,
    "GEMM_003": gen_gemm_003,
    "GEMM_004": gen_gemm_004,
    "MATMUL_001": gen_matmul_001,
    "FLATTEN_001": gen_flatten_001,
    "RESHAPE_001": gen_reshape_001,
    "SQUEEZE_001": gen_squeeze_001,
    "CONV_001": gen_conv_001,
    "CONV_002": gen_conv_002,
    "CONV_003": gen_conv_003,
    "CONV_004": gen_conv_004,
    "MAXPOOL_001": gen_maxpool_001,
    "MAXPOOL_002": gen_maxpool_002,
    "AVGPOOL_001": gen_avgpool_001,
    "AVGPOOL_002": gen_avgpool_002,
    "AVGPOOL_003": gen_avgpool_003,
    "SOFTMAX_001": gen_softmax_001,
    "SOFTMAX_002": gen_softmax_002,
    "CONCAT_001": gen_concat_001,
    "CONCAT_002": gen_concat_002,
    "ADD_001": gen_add_001,
    "MUL_001": gen_mul_001,
    "SUB_001": gen_sub_001,
    "DIV_001": gen_div_001,
    "SIGMOID_001": gen_sigmoid_001,
    "TANH_001": gen_tanh_001,
    "LEAKYRELU_001": gen_leakyrelu_001,
    "CLIP_001": gen_clip_001,
    "NEGOP_001": gen_negop_001,
    "SQRT_001": gen_sqrt_001,
    "RECIPROCAL_001": gen_reciprocal_001,
    "REDUCEMEAN_001": gen_reducemean_001,
    "UNSQUEEZE_001": gen_unsqueeze_001,
    "EXP_001": gen_exp_001,
    "LOG_001": gen_log_001,
    "FLOOR_001": gen_floor_001,
    "CEIL_001": gen_ceil_001,
    "ROUND_001": gen_round_001,
    "SIGN_001": gen_sign_001,
    "MIN_001": gen_min_001,
    "MAX_001": gen_max_001,
    "POW_001": gen_pow_001,
    "REDUCESUM_001": gen_reducesum_001,
    "REDUCEMAX_001": gen_reducemax_001,
    "REDUCEMIN_001": gen_reducemin_001,
    "WHERE_001": gen_where_001,
    "EQUAL_001": gen_equal_001,
    "GREATER_001": gen_greater_001,
    "LESS_001": gen_less_001,
    "GREATEROREQUAL_001": gen_greaterorequal_001,
    "LESSOREQUAL_001": gen_lessorequal_001,
    "AND_001": gen_and_001,
    "OR_001": gen_or_001,
    "NOT_001": gen_not_001,
    "XOR_001": gen_xor_001,
    "ERF_001": gen_erf_001,
    "SOFTPLUS_001": gen_softplus_001,
    "SOFTSIGN_001": gen_softsign_001,
    "HARDSWISH_001": gen_hardswish_001,
    "ELU_001": gen_elu_001,
    "SELU_001": gen_selu_001,
    "HARDSIGMOID_001": gen_hardsigmoid_001,
    "THRESHOLDEDRELU_001": gen_thresholdedrelu_001,
    "CELU_001": gen_celu_001,
    "PRELU_001": gen_prelu_001,
    "REDUCEPROD_001": gen_reduceprod_001,
    "REDUCEL1_001": gen_reducel1_001,
    "REDUCEL2_001": gen_reducel2_001,
    "REDUCELOGSUM_001": gen_reducelogsum_001,
    "REDUCELOGSUMEXP_001": gen_reducelogsumexp_001,
    "REDUCESUMSQUARE_001": gen_reducesumsquare_001,
    "GLOBALMAXPOOL_001": gen_globalmaxpool_001,
    "GLOBALLPPOOL_001": gen_globallppool_001,
    "LPPOOL_001": gen_lppool_001,
    "LPNORMALIZATION_001": gen_lpnormalization_001,
    "CUMSUM_001": gen_cumsum_001,
    "MEAN_001": gen_mean_001,
    "SUM_001": gen_sum_001,
    "IDENTITY_001": gen_identity_001,
    "CAST_001": gen_cast_001,
    "PAD_001": gen_pad_001,
    "SLICE_001": gen_slice_001,
    "GATHER_001": gen_gather_001,
    "TRANSPOSE_001": gen_transpose_001,
    "TOPO_001": gen_topo_001,
    "TOPO_002": gen_topo_002,
    "TOPO_003": gen_topo_003,
    "NET_001": gen_net_001,
    "NET_002": gen_net_002,
    "NET_003": gen_net_003,
    "NET_004": gen_net_004,
    "NET_005": gen_net_005,
    "NET_006": gen_net_006,
    "TS_001": lambda: _copy_fixture_model("TS_001", "timeseries_cwru_bearing_mlp.onnx"),
    "TS_002": lambda: _copy_fixture_model("TS_002", "timeseries_kws_dscnn_small_int8.onnx"),
    "TS_003": lambda: _copy_fixture_model("TS_003", "timeseries_kws_dscnn_small_qat_int8.onnx"),
    "TS_004": lambda: _copy_fixture_model("TS_004", "timeseries_met_hybrid.onnx"),
    "TS_005": lambda: _copy_fixture_model("TS_005", "timeseries_stwin_vowel.onnx"),
    "QLINEAR_NUM_001": gen_qlinear_num_001,
    "QLINEAR_NUM_002": gen_qlinear_num_002,
    "QLINEAR_NUM_003": gen_qlinear_num_003,
    "NEG_001": gen_neg_001,
    "NEG_002": gen_neg_002,
    "NEG_003": gen_neg_003,
}



def _check_registry_sync() -> None:
    """启动时校验 _GENERATORS 与 CASE_MAP 是否一致，防止漏注册或多注册。"""
    registry_ids = set(CASE_MAP.keys())
    generator_ids = set(_GENERATORS.keys())
    missing = registry_ids - generator_ids
    extra = generator_ids - registry_ids
    if missing:
        print(f"[警告] cases_registry 中有用例但缺少生成函数: {sorted(missing)}")
    if extra:
        print(f"[警告] _GENERATORS 中有生成函数但 cases_registry 无对应记录: {sorted(extra)}")


def main() -> None:
    _check_registry_sync()

    parser = argparse.ArgumentParser(description="TDD 测试模型生成器")
    parser.add_argument("--case", type=str, default=None, help="生成单个用例 (如 GEMM_001)")
    parser.add_argument("--category", type=str, default=None, help="生成某一类 (如 core/gemm)")
    args = parser.parse_args()

    if args.case:
        if args.case not in _GENERATORS:
            print(f"未知用例: {args.case}")
            print(f"可用: {', '.join(sorted(_GENERATORS.keys()))}")
            sys.exit(1)
        path = _GENERATORS[args.case]()
        print(f"  生成: {path}")
    elif args.category:
        matched = {
            k: v
            for k, v in _GENERATORS.items()
            if CASE_MAP[k].category == args.category
        }
        if not matched:
            print(f"无匹配用例: {args.category}")
            sys.exit(1)
        for _case_id, gen_fn in sorted(matched.items()):
            path = gen_fn()
            print(f"  生成: {path}")
    else:
        print(f"生成全部 {len(_GENERATORS)} 个测试模型...")
        for case_id, gen_fn in sorted(_GENERATORS.items()):
            path = gen_fn()
            print(f"  [{case_id}] {path}")
        print(f"\n完成。共 {len(_GENERATORS)} 个模型。")


if __name__ == "__main__":
    main()
