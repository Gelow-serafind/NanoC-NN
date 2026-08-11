# MEAN_001: 官方 Mean QDQ/int8 双输入逐元素均值

## 验证目标

验证 `Mean` 对两个同形状 QDQ/int8 tensor 生成 C99 逐元素均值路径，输出与 ONNX Runtime 数值一致。

## 来源

内部探索：ONNX 官方图谱能力扩展，100% 计划推进。

## ONNX Schema 归属

| 字段 | 值 |
|------|----|
| domain | `ai.onnx` |
| op_type | `Mean` |
| opset_range | `1+` |
| schema_form | `two same-shape QDQ/int8 tensors, elementwise mean` |
| lowering | `generated_c_mean_s8` |
| backend | `cmsis-nn` |

## 网络结构

```text
Input -> Q/DQ
Static QDQ rhs + input_dq -> Mean -> Q/DQ Output
```

## 输入

- **张量形状**: `[1, 8]`
- **数据类型**: int8 QDQ 数据路径

## 算子参数

见 ONNX Schema 归属表与网络结构，采用默认/首个子形态参数。

## 量化设计

| 张量 | scale | zero_point |
|------|-------|------------|
| input | `0.05` | `0` |
| rhs | `0.05` | `0` |
| output | `0.05` | `0` |

## 预期结果

- **codegen status**: `ok`
- **数值验收**: `top1=2/2`、最大绝对误差 `0.05`、饱和率 ≤ `0.25`
- **数据集**: `tdd/fixtures/datasets/mean_qdq_smoke/dataset.json`

## 边界/风险

首个形态固定双输入（第二输入为量化常量）；除法 `/2` 存在 1-ULP 差异，允许 `max_abs=0.05`。
