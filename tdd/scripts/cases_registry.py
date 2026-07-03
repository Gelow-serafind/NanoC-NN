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
class CaseDef:
    case_id: str
    description: str
    expected: str
    category: str
    required_apis: tuple[str, ...] = field(default_factory=tuple)
    regression: bool = False


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
    CaseDef("NEG_001", "float32 无 Q/DQ 模型正确拒绝", "blocked", "negative"),
]

CASE_MAP: dict[str, CaseDef] = {c.case_id: c for c in CASE_REGISTRY}
