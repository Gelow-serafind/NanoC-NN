"""
tdd/scripts/cases_registry.py — 测试用例元数据统一注册表

这是用例配置的唯一来源（Single Source of Truth）。
run_tests.py 和 generate_models.py 都从这里读取，不再各自维护分散的字典。

新增用例时只需在 CASE_REGISTRY 里加一条记录；
生成函数在 generate_models.py 的 _GENERATORS 里补充同名条目。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NumericCheck:
    dataset_id: str
    input_scale: float = 255.0
    top1_min_match_ratio: float = 1.0
    max_abs_error: float | None = None
    max_saturation_ratio: float = 0.5


@dataclass(frozen=True)
class CaseDef:
    case_id: str
    description: str
    expected: str
    category: str
    required_apis: tuple[str, ...] = field(default_factory=tuple)
    regression: bool = False
    numeric: NumericCheck | None = None


CASE_REGISTRY: list[CaseDef] = [
    CaseDef(
        "GEMM_001",
        "最小对称 FC 无 bias",
        "ok",
        "core/gemm",
        ("arm_fully_connected_s8",),
    ),
    CaseDef(
        "GEMM_002",
        "非对称输入 zp!=0 含 bias",
        "ok",
        "core/gemm",
        ("arm_fully_connected_s8",),
    ),
    CaseDef(
        "GEMM_003",
        "非 4 对齐维度 13->7",
        "ok",
        "core/gemm",
        ("arm_fully_connected_s8",),
    ),
    CaseDef(
        "GEMM_004",
        "中等规模 64->32",
        "ok",
        "core/gemm",
        ("arm_fully_connected_s8",),
    ),
    CaseDef(
        "CONV_001",
        "1x1 pointwise 单通道",
        "ok",
        "core/conv",
        ("arm_convolve_wrapper_s8",),
    ),
    CaseDef(
        "CONV_002",
        "3x3 标准 SAME padding",
        "ok",
        "core/conv",
        ("arm_convolve_wrapper_s8",),
    ),
    CaseDef(
        "CONV_003",
        "stride=2 下采样",
        "ok",
        "core/conv",
        ("arm_convolve_wrapper_s8",),
    ),
    CaseDef(
        "CONV_004",
        "非对称输入 zp!=0",
        "ok",
        "core/conv",
        ("arm_convolve_wrapper_s8",),
    ),
    CaseDef(
        "MAXPOOL_001",
        "标准 2x2 stride=2 MaxPool int8 代码生成",
        "ok",
        "core/maxpool",
        ("arm_max_pool_s8",),
        regression=True,
    ),
    CaseDef(
        "SOFTMAX_001",
        "10 分类 Softmax int8 代码生成",
        "ok",
        "core/softmax",
        ("arm_softmax_s8",),
        regression=True,
    ),
    CaseDef(
        "TOPO_001",
        "Conv->Relu->Pool->Flatten->FC 组合链路",
        "ok",
        "topology",
        ("arm_convolve_wrapper_s8", "arm_max_pool_s8", "arm_fully_connected_s8"),
    ),
    CaseDef(
        "TOPO_002",
        "Conv1d->Relu->Conv1d->Relu->Flatten->FC 时序分类链路",
        "ok",
        "topology",
        ("arm_convolve_wrapper_s8", "arm_fully_connected_s8"),
    ),
    CaseDef(
        "TOPO_003",
        "真实 MNIST QLinear int8 分类链路",
        "ok",
        "topology",
        (
            "arm_convolve_wrapper_s8",
            "arm_max_pool_s8",
            "arm_fully_connected_per_channel_s8",
            "arm_elementwise_add_s8",
        ),
        regression=True,
        numeric=NumericCheck(
            dataset_id="mnist_synthetic_smoke",
            input_scale=255.0,
            top1_min_match_ratio=1.0,
            max_saturation_ratio=0.5,
        ),
    ),
    CaseDef(
        "NET_001",
        "真实 SqueezeNet 1.0 int8 图像分类网络导入评估",
        "blocked",
        "networks",
        regression=False,
    ),
    CaseDef(
        "NET_002",
        "真实 MobileNetV2 int8/QLinear 图像分类网络导入评估",
        "ok",
        "networks",
        (
            "arm_convolve_wrapper_s8",
            "arm_depthwise_conv_wrapper_s8",
            "arm_elementwise_add_s8",
            "arm_avgpool_s8",
            "arm_fully_connected_per_channel_s8",
        ),
        regression=False,
    ),
    CaseDef(
        "NET_003",
        "真实 SSD-MobileNet int8 目标检测网络导入评估",
        "unsupported",
        "networks",
        regression=False,
    ),
    CaseDef(
        "NET_004",
        "KWS DS-CNN-style int8 网络导入评估",
        "ok",
        "networks",
        (
            "arm_convolve_wrapper_s8",
            "arm_depthwise_conv_wrapper_s8",
            "arm_max_pool_s8",
            "arm_fully_connected_s8",
        ),
        regression=False,
    ),
    CaseDef(
        "NET_005",
        "Tiny signal jump int8 时序分类网络导入评估",
        "ok",
        "networks",
        (
            "arm_convolve_wrapper_s8",
            "arm_fully_connected_s8",
        ),
        regression=True,
        numeric=NumericCheck(
            dataset_id="net_005_signal_jump_smoke",
            input_scale=1.0,
            top1_min_match_ratio=1.0,
            max_saturation_ratio=0.5,
        ),
    ),
    CaseDef(
        "NET_006",
        "EfficientNet-Lite4 int8 小型图像分类网络导入评估",
        "unsupported",
        "networks",
        regression=False,
    ),
    CaseDef(
        "QLINEAR_NUM_001",
        "QLinearConv uint8 输入数值精度 (CMSIS-NN vs ONNX)",
        "ok",
        "core/qlinear",
        ("arm_convolve_wrapper_s8",),
    ),
    CaseDef(
        "QLINEAR_NUM_002",
        "多通道 QLinearConv + bias + per-channel scale",
        "ok",
        "core/qlinear",
        ("arm_convolve_wrapper_s8",),
    ),
    CaseDef(
        "QLINEAR_NUM_003",
        "高通道 QLinearConv 5×5 per-channel (仿 MNIST Conv1)",
        "ok",
        "core/qlinear",
        ("arm_convolve_wrapper_s8",),
    ),
    CaseDef("NEG_001", "float32 无 Q/DQ 模型正确拒绝", "blocked", "negative"),
]

CASE_MAP: dict[str, CaseDef] = {c.case_id: c for c in CASE_REGISTRY}
