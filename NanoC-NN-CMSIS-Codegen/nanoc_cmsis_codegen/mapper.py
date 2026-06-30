from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpMapping:
    onnx_op: str
    status: str
    cmsis_action: str
    reason: str = ""


INITIAL_OP_MAPPING = {
    "Add": OpMapping("Add", "direct_api", "arm_elementwise_add_s8"),
    "AveragePool": OpMapping("AveragePool", "direct_api", "arm_avgpool_s8"),
    "BatchNormalization": OpMapping("BatchNormalization", "folded", "fold into Conv/FC parameters"),
    "Clip": OpMapping("Clip", "fused", "cmsis_nn_activation min/max"),
    "Concat": OpMapping("Concat", "direct_api", "arm_concatenation_s8_x/y/z/w"),
    "Conv": OpMapping("Conv", "wrapper_api", "arm_convolve_wrapper_s8"),
    "Flatten": OpMapping("Flatten", "folded", "shape-only generation step"),
    "Gemm": OpMapping("Gemm", "wrapper_api", "arm_fully_connected_wrapper_s8"),
    "GlobalAveragePool": OpMapping("GlobalAveragePool", "direct_api", "arm_avgpool_s8"),
    "MatMul": OpMapping("MatMul", "wrapper_api", "arm_fully_connected_wrapper_s8"),
    "MaxPool": OpMapping("MaxPool", "direct_api", "arm_max_pool_s8"),
    "Mul": OpMapping("Mul", "direct_api", "arm_elementwise_mul_s8"),
    "Relu": OpMapping("Relu", "fused", "cmsis_nn_activation min/max"),
    "Reshape": OpMapping("Reshape", "folded", "shape-only generation step"),
    "Softmax": OpMapping("Softmax", "direct_api", "arm_softmax_s8"),
    "Transpose": OpMapping("Transpose", "direct_api", "arm_transpose_s8"),
}
