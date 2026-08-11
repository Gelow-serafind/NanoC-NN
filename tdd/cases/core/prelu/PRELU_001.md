# PRELU_001: 官方 PRelu QDQ/int8 常量 slope

## 验证目标

验证 `PRelu` 对同形状 QDQ/int8 tensor 与量化常量 slope 生成 C99 逐元素带泄漏路径，输出与 ONNX Runtime 数值一致。

## 来源

内部探索：ONNX 官方图谱二元激活扩展，slope 作为量化常量第二输入，100% 计划 W1b。

## ONNX Schema 归属

| 字段 | 值 |
|------|----|
| domain | `ai.onnx` |
| op_type | `PRelu` |
| opset_range | `1+` |
| schema_form | `same-shape QDQ/int8 tensor, constant quantized slope input` |
| lowering | `generated_c_prelu_s8` |
| backend | `cmsis-nn` |

## 网络结构

```text
Input -> Q/DQ
Static QDQ slope + input_dq -> PRelu -> Q/DQ Output
```

## 输入

- **张量形状**: `[1, 8]`
- **数据类型**: int8 QDQ 数据路径，输入域 `[-4, 4]`，slope 为量化常量

## 算子参数

| 算子 | 参数 |
|------|------|
| PRelu | 逐元素 `x>0 ? x : slope*x` |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|------------|------|
| input | `0.05` | `0` | 外部输入量化 |
| slope | `0.05` | `0` | 量化常量 slope（int8） |
| output | `0.05` | `0` | 输出重新量化 |

## 预期结果

- **codegen status**: `ok`
- **若 ok**: 编译通过、运行不崩溃
- **关键验证点**: 生成 C 中存在逐元素 slope 反量化乘法

## 数值验收

- **是否需要**: `yes`
- **数据集**: `tdd/fixtures/datasets/prelu_qdq_smoke/dataset.json`
- **参考路径**: 原始 ONNX Runtime
- **通过阈值**: top1 一致率 `1.0`，最大绝对误差 `0.05`，饱和率不超过 `0.25`

## 边界/风险

当前只验证同形状常量 slope，不覆盖广播 slope、外部运行时 slope 输入或负 slope。
