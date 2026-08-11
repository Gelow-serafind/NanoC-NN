# HARDSIGMOID_001: 官方 HardSigmoid QDQ/int8 静态张量

## 验证目标

验证 `HardSigmoid` 对静态 QDQ/int8 tensor 生成 C99 逐元素 hard sigmoid 路径（含 `alpha`/`beta` 属性），输出与 ONNX Runtime 数值一致。

## 来源

内部探索：ONNX 官方图谱属性 unary 激活扩展，100% 计划 W1b。

## ONNX Schema 归属

| 字段 | 值 |
|------|----|
| domain | `ai.onnx` |
| op_type | `HardSigmoid` |
| opset_range | `1+` |
| schema_form | `static QDQ/int8 tensor, alpha/beta attributes` |
| lowering | `generated_c_hardsigmoid_s8` |
| backend | `cmsis-nn` |

## 网络结构

```text
Input -> Q/DQ -> HardSigmoid(alpha=0.2, beta=0.5) -> Q/DQ Output
```

## 输入

- **张量形状**: `[1, 8]`
- **数据类型**: int8 QDQ 数据路径，输入域 `[-6, 6]`

## 算子参数

| 算子 | 参数 |
|------|------|
| HardSigmoid | `alpha=0.2`、`beta=0.5`，逐元素 `clamp(alpha*x+beta, 0, 1)` |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|------------|------|
| input | `0.05` | `0` | 外部输入量化 |
| output | `0.05` | `0` | 输出重新量化 |

## 预期结果

- **codegen status**: `ok`
- **若 ok**: 编译通过、运行不崩溃
- **关键验证点**: 生成 C 中存在逐元素 clamp 与 alpha/beta 属性

## 数值验收

- **是否需要**: `yes`
- **数据集**: `tdd/fixtures/datasets/hardsigmoid_qdq_smoke/dataset.json`
- **参考路径**: 原始 ONNX Runtime
- **通过阈值**: top1 一致率 `1.0`，最大绝对误差 `0.05`，饱和率不超过 `0.25`

## 边界/风险

输出有界于 `[0, 1]`，量化后不饱和；`alpha*x+beta` 存在 1-ULP 级差异可能，允许最大绝对误差 `0.05`。
