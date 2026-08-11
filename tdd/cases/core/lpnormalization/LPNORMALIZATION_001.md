# LPNORMALIZATION_001: 官方 LpNormalization QDQ/int8 行归一化

## 验证目标

验证 `LpNormalization` 对 rank=2 静态 QDQ/int8 tensor 生成 C99 行 Lp 归一化路径，输出与 ONNX Runtime 数值一致。

## 来源

内部探索：ONNX 官方图谱能力扩展，100% 计划推进。

## ONNX Schema 归属

| 字段 | 值 |
|------|----|
| domain | `ai.onnx` |
| op_type | `LpNormalization` |
| opset_range | `1+` |
| schema_form | `rank=2 static QDQ/int8 tensor, axis=1, p=2 row Lp normalization` |
| lowering | `generated_c_lpnormalization_s8` |
| backend | `cmsis-nn` |

## 网络结构

```text
Input [2,4] -> Q/DQ -> LpNormalization(axis=1, p=2) -> [2,4] -> Q/DQ Output
```

## 输入

- **张量形状**: `[2,4]` -> `[2,4]`
- **数据类型**: int8 QDQ 数据路径

## 算子参数

见 ONNX Schema 归属表与网络结构，采用默认/首个子形态参数。

## 量化设计

| 张量 | scale | zero_point |
|------|-------|------------|
| input | `0.05` | `0` |
| output | `0.05` | `0` |

## 预期结果

- **codegen status**: `ok`
- **数值验收**: `top1=2/2`、最大绝对误差 `0.05`、饱和率 ≤ `0.25`
- **数据集**: `tdd/fixtures/datasets/lpnormalization_qdq_smoke/dataset.json`

## 边界/风险

除法存在 1-ULP 差异，允许 `max_abs=0.05`；行范数为 0 时输出 0。
