# CELU_001: 官方 Celu QDQ/int8 静态张量

## 验证目标

验证 `Celu` 对静态 QDQ/int8 tensor 生成 C99 逐元素 celu 路径（含 `alpha` 属性），输出与 ONNX Runtime 数值一致。

## 来源

内部探索：ONNX 官方图谱属性 unary 激活扩展，100% 计划 W1b。

## ONNX Schema 归属

| 字段 | 值 |
|------|----|
| domain | `ai.onnx` |
| op_type | `Celu` |
| opset_range | `12+` |
| schema_form | `static QDQ/int8 tensor, alpha attribute` |
| lowering | `generated_c_celu_s8` |
| backend | `cmsis-nn` |

## 网络结构

```text
Input -> Q/DQ -> Celu(alpha=1.0) -> Q/DQ Output
```

## 输入

- **张量形状**: `[1, 8]`
- **数据类型**: int8 QDQ 数据路径，输入域 `[-5, 5]`

## 算子参数

| 算子 | 参数 |
|------|------|
| Celu | `alpha=1.0`，逐元素 `max(0,x)+min(0,alpha*(exp(x/alpha)-1))` |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|------------|------|
| input | `0.05` | `0` | 外部输入量化 |
| output | `0.05` | `0` | 输出重新量化 |

## 预期结果

- **codegen status**: `ok`
- **若 ok**: 编译通过、运行不崩溃
- **关键验证点**: 生成 C 中存在逐元素 `expf` 分支与 alpha 属性

## 数值验收

- **是否需要**: `yes`
- **数据集**: `tdd/fixtures/datasets/celu_qdq_smoke/dataset.json`
- **参考路径**: 原始 ONNX Runtime
- **通过阈值**: top1 一致率 `1.0`，最大绝对误差 `0.05`，饱和率不超过 `0.25`

## 边界/风险

`expf` 存在 1-ULP 级差异可能，允许最大绝对误差 `0.05`；输入域受控避免负向 exp 饱和。
