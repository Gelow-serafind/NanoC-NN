# THRESHOLDEDRELU_001: 官方 ThresholdedRelu QDQ/int8 静态张量

## 验证目标

验证 `ThresholdedRelu` 对静态 QDQ/int8 tensor 生成 C99 逐元素 thresholded relu 路径（含 `alpha` 属性），输出与 ONNX Runtime 数值一致。

## 来源

内部探索：ONNX 官方图谱属性 unary 激活扩展，100% 计划 W1b。

## ONNX Schema 归属

| 字段 | 值 |
|------|----|
| domain | `ai.onnx` |
| op_type | `ThresholdedRelu` |
| opset_range | `1+` |
| schema_form | `static QDQ/int8 tensor, alpha attribute` |
| lowering | `generated_c_thresholdedrelu_s8` |
| backend | `cmsis-nn` |

## 网络结构

```text
Input -> Q/DQ -> ThresholdedRelu(alpha=1.0) -> Q/DQ Output
```

## 输入

- **张量形状**: `[1, 8]`
- **数据类型**: int8 QDQ 数据路径，输入域 `[-2, 4]`

## 算子参数

| 算子 | 参数 |
|------|------|
| ThresholdedRelu | `alpha=1.0`，逐元素 `x>alpha ? x : 0` |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|------------|------|
| input | `0.05` | `0` | 外部输入量化 |
| output | `0.05` | `0` | 输出重新量化 |

## 预期结果

- **codegen status**: `ok`
- **若 ok**: 编译通过、运行不崩溃
- **关键验证点**: 生成 C 中存在逐元素阈值比较与 alpha 属性

## 数值验收

- **是否需要**: `yes`
- **数据集**: `tdd/fixtures/datasets/thresholdedrelu_qdq_smoke/dataset.json`
- **参考路径**: 原始 ONNX Runtime
- **通过阈值**: top1 一致率 `1.0`，最大绝对误差 `0.0`，饱和率不超过 `0.25`

## 边界/风险

`x>alpha ? x : 0` 为精确分支运算，无超越函数，期望 `max_abs=0.0`。
