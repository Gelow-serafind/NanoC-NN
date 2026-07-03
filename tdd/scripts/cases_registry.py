"""
tdd/scripts/cases_registry.py — 测试用例元数据统一注册表

这是用例配置的唯一来源（Single Source of Truth）。
run_tests.py 和 generate_models.py 都从这里读取，不再各自维护分散的字典。

新增用例时只需在 CASE_REGISTRY 里加一条记录；
生成函数在 generate_models.py 的 _GENERATORS 里补充同名条目。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CaseDef:
    case_id: str
    description: str
    expected: str


CASE_REGISTRY: list[CaseDef] = [
    CaseDef("GEMM_001", "最小对称 FC 无 bias",          "ok"),
    CaseDef("GEMM_002", "非对称输入 zp!=0 含 bias",     "ok"),
    CaseDef("GEMM_003", "非 4 对齐维度 13->7",          "ok"),
    CaseDef("GEMM_004", "中等规模 64->32",               "ok"),
    CaseDef("CONV_001", "1x1 pointwise 单通道",          "ok"),
    CaseDef("CONV_002", "3x3 标准 SAME padding",         "ok"),
    CaseDef("CONV_003", "stride=2 下采样",               "ok"),
    CaseDef("CONV_004", "非对称输入 zp!=0",              "ok"),
    CaseDef("MAXPOOL_001", "标准 2x2 stride=2 MaxPool int8 代码生成", "ok"),
    CaseDef("SOFTMAX_001", "10 分类 Softmax int8 代码生成",           "ok"),
    CaseDef("TOPO_001",  "Conv->Relu->Pool->Flatten->FC 组合链路",    "ok"),
    CaseDef("NEG_001",   "float32 无 Q/DQ 模型正确拒绝",              "unsupported"),
]

CASE_MAP: dict[str, CaseDef] = {c.case_id: c for c in CASE_REGISTRY}
