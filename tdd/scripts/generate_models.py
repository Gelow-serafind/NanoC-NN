"""
tdd/scripts/generate_models.py — 根据 cases/ 规格批量生成 ONNX 测试模型

用法:
    python tdd/scripts/generate_models.py
    python tdd/scripts/generate_models.py --case GEMM_001
    python tdd/scripts/generate_models.py --category core/gemm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from cases_registry import CASE_MAP  # noqa: E402
from common.model_builder import (  # noqa: E402
    build_and_save,
    make_conv_node,
    make_flatten_node,
    make_gemm_node,
    make_maxpool_node,
    make_qdq_wrapper,
    make_relu_node,
    make_softmax_node,
)

TDD_ROOT = _SCRIPT_DIR.parent
MODELS_ROOT = TDD_ROOT / "models"


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
    )


# ---------------------------------------------------------------------------
# GEMM 用例生成
# ---------------------------------------------------------------------------

def gen_gemm_001() -> Path:
    """GEMM_001: 最小对称 FC 无 bias — [1,2] → [1,3]"""
    path = MODELS_ROOT / "core" / "gemm" / "GEMM_001.onnx"
    gemm_node, gemm_inits = make_gemm_node(
        "gemm", ["input_dq"], ["output"], weight_shape=[3, 2], with_bias=False
    )
    _build_qdq_model(
        graph_name="gemm_001",
        input_shape=[1, 2], input_scale=0.01, input_zp=0,
        output_shape=[1, 3], output_scale=0.05, output_zp=0,
        runtime_nodes=[gemm_node],
        runtime_initializers=gemm_inits,
        save_path=path,
    )
    return path


def gen_gemm_002() -> Path:
    """GEMM_002: 非对称输入 zp=10 含 bias — [1,4] → [1,8]"""
    path = MODELS_ROOT / "core" / "gemm" / "GEMM_002.onnx"
    gemm_node, gemm_inits = make_gemm_node(
        "gemm", ["input_dq"], ["output"], weight_shape=[8, 4], with_bias=True
    )
    _build_qdq_model(
        graph_name="gemm_002",
        input_shape=[1, 4], input_scale=0.01, input_zp=10,
        output_shape=[1, 8], output_scale=0.05, output_zp=0,
        runtime_nodes=[gemm_node],
        runtime_initializers=gemm_inits,
        save_path=path,
    )
    return path


def gen_gemm_003() -> Path:
    """GEMM_003: 非 4 对齐维度 13→7"""
    path = MODELS_ROOT / "core" / "gemm" / "GEMM_003.onnx"
    gemm_node, gemm_inits = make_gemm_node(
        "gemm", ["input_dq"], ["output"], weight_shape=[7, 13], with_bias=True
    )
    _build_qdq_model(
        graph_name="gemm_003",
        input_shape=[1, 13], input_scale=0.03, input_zp=0,
        output_shape=[1, 7], output_scale=0.04, output_zp=0,
        runtime_nodes=[gemm_node],
        runtime_initializers=gemm_inits,
        save_path=path,
    )
    return path


def gen_gemm_004() -> Path:
    """GEMM_004: 中等规模 64→32"""
    path = MODELS_ROOT / "core" / "gemm" / "GEMM_004.onnx"
    gemm_node, gemm_inits = make_gemm_node(
        "gemm", ["input_dq"], ["output"], weight_shape=[32, 64], with_bias=True
    )
    _build_qdq_model(
        graph_name="gemm_004",
        input_shape=[1, 64], input_scale=0.01, input_zp=0,
        output_shape=[1, 32], output_scale=0.02, output_zp=0,
        runtime_nodes=[gemm_node],
        runtime_initializers=gemm_inits,
        save_path=path,
    )
    return path


# ---------------------------------------------------------------------------
# CONV 用例生成
# ---------------------------------------------------------------------------

def gen_conv_001() -> Path:
    """CONV_001: 1×1 pointwise 单通道 — [1,1,4,4] → [1,1,4,4]"""
    path = MODELS_ROOT / "core" / "conv" / "CONV_001.onnx"
    conv_node, conv_inits = make_conv_node(
        "conv", ["input_dq"], ["output"],
        weight_shape=[1, 1, 1, 1], kernel_shape=[1, 1],
        strides=[1, 1], pads=[0, 0, 0, 0], with_bias=True,
    )
    _build_qdq_model(
        graph_name="conv_001",
        input_shape=[1, 1, 4, 4], input_scale=0.01, input_zp=0,
        output_shape=[1, 1, 4, 4], output_scale=0.05, output_zp=0,
        runtime_nodes=[conv_node],
        runtime_initializers=conv_inits,
        save_path=path,
    )
    return path


def gen_conv_002() -> Path:
    """CONV_002: 3×3 标准 SAME padding — [1,3,8,8] → [1,8,8,8]"""
    path = MODELS_ROOT / "core" / "conv" / "CONV_002.onnx"
    conv_node, conv_inits = make_conv_node(
        "conv", ["input_dq"], ["output"],
        weight_shape=[8, 3, 3, 3], kernel_shape=[3, 3],
        strides=[1, 1], pads=[1, 1, 1, 1], with_bias=True,
    )
    _build_qdq_model(
        graph_name="conv_002",
        input_shape=[1, 3, 8, 8], input_scale=0.01, input_zp=0,
        output_shape=[1, 8, 8, 8], output_scale=0.05, output_zp=0,
        runtime_nodes=[conv_node],
        runtime_initializers=conv_inits,
        save_path=path,
    )
    return path


def gen_conv_003() -> Path:
    """CONV_003: stride=2 下采样 — [1,4,8,8] → [1,8,4,4]"""
    path = MODELS_ROOT / "core" / "conv" / "CONV_003.onnx"
    conv_node, conv_inits = make_conv_node(
        "conv", ["input_dq"], ["output"],
        weight_shape=[8, 4, 3, 3], kernel_shape=[3, 3],
        strides=[2, 2], pads=[1, 1, 1, 1], with_bias=True,
    )
    _build_qdq_model(
        graph_name="conv_003",
        input_shape=[1, 4, 8, 8], input_scale=0.01, input_zp=0,
        output_shape=[1, 8, 4, 4], output_scale=0.03, output_zp=0,
        runtime_nodes=[conv_node],
        runtime_initializers=conv_inits,
        save_path=path,
    )
    return path


def gen_conv_004() -> Path:
    """CONV_004: 非对称输入 zp=12 — [1,3,8,8] → [1,8,8,8]"""
    path = MODELS_ROOT / "core" / "conv" / "CONV_004.onnx"
    conv_node, conv_inits = make_conv_node(
        "conv", ["input_dq"], ["output"],
        weight_shape=[8, 3, 3, 3], kernel_shape=[3, 3],
        strides=[1, 1], pads=[1, 1, 1, 1], with_bias=True,
    )
    _build_qdq_model(
        graph_name="conv_004",
        input_shape=[1, 3, 8, 8], input_scale=0.01, input_zp=12,
        output_shape=[1, 8, 8, 8], output_scale=0.05, output_zp=0,
        runtime_nodes=[conv_node],
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

    conv_node, conv_inits = make_conv_node(
        "conv", ["input_dq"], ["conv_out"],
        weight_shape=[8, 3, 3, 3], kernel_shape=[3, 3],
        strides=[1, 1], pads=[1, 1, 1, 1], with_bias=True,
    )
    nodes.append(conv_node)
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

    gemm_node, gemm_inits = make_gemm_node(
        "fc", ["flat_out_dq_dq"], ["fc_out"], weight_shape=[10, 128], with_bias=True
    )
    nodes.append(gemm_node)
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
    )
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


# ---------------------------------------------------------------------------
# 生成函数映射（必须与 cases_registry.CASE_MAP 中的 case_id 保持一一对应）
# ---------------------------------------------------------------------------

_GENERATORS: dict[str, object] = {
    "GEMM_001": gen_gemm_001,
    "GEMM_002": gen_gemm_002,
    "GEMM_003": gen_gemm_003,
    "GEMM_004": gen_gemm_004,
    "CONV_001": gen_conv_001,
    "CONV_002": gen_conv_002,
    "CONV_003": gen_conv_003,
    "CONV_004": gen_conv_004,
    "MAXPOOL_001": gen_maxpool_001,
    "SOFTMAX_001": gen_softmax_001,
    "TOPO_001": gen_topo_001,
    "NEG_001": gen_neg_001,
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
        prefix = args.category.split("/")[-1].upper()
        matched = {k: v for k, v in _GENERATORS.items() if k.startswith(prefix)}
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
