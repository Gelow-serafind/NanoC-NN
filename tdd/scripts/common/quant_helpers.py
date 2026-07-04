"""
tdd/scripts/common/quant_helpers.py — 量化参数辅助函数

提供 Q/DQ 量化 scale/zp 的常用组合和校验工具。
"""

from __future__ import annotations

from typing import NamedTuple


class QuantParams(NamedTuple):
    """量化参数组。"""
    scale: float
    zero_point: int
    symmetric: bool  # zero_point == 0

    @property
    def is_symmetric(self) -> bool:
        return self.zero_point == 0


# 预定义量化参数组合
SYMMETRIC_DEFAULT = QuantParams(scale=0.01, zero_point=0, symmetric=True)
SYMMETRIC_LARGE = QuantParams(scale=0.05, zero_point=0, symmetric=True)
SYMMETRIC_SMALL = QuantParams(scale=0.001, zero_point=0, symmetric=True)

ASYMMETRIC_ZP10 = QuantParams(scale=0.01, zero_point=10, symmetric=False)
ASYMMETRIC_ZP_NEG5 = QuantParams(scale=0.03, zero_point=-5, symmetric=False)
ASYMMETRIC_ZP5 = QuantParams(scale=0.02, zero_point=5, symmetric=False)


def make_params(
    scale: float = 0.01,
    zero_point: int = 0,
) -> QuantParams:
    """工厂函数：创建量化参数。"""
    return QuantParams(scale=scale, zero_point=zero_point, symmetric=(zero_point == 0))


def params_match_for_pool(
    input_params: QuantParams,
    output_params: QuantParams,
) -> bool:
    """检查 MaxPool/AveragePool 所需的输入输出量化一致性。

    CMSIS-NN s8 Pooling 要求输入和输出的 scale 和 zero_point 完全相同。
    """
    return (
        input_params.scale == output_params.scale
        and input_params.zero_point == output_params.zero_point
    )
