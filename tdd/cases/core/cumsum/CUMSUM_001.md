# CUMSUM_001: 官方 CumSum QDQ/int8 轴累计和

## 验证目标

验证 `CumSum` 对 rank=2 静态 QDQ/int8 tensor 生成 C99 轴累计和路径，输出与 ONNX Runtime 数值一致。

## 来源

内部探索：ONNX 官方图谱能力扩展，100% 计划推进。

## ONNX Schema 归属

| 字段 | 值 |
|------|----|
| domain | `ai.onnx` |
| op_type | `CumSum` |
| opset_range | `11+` |
| schema_form | `rank=2 static QDQ/int8 tensor, axis=1 exclusive=0 reverse=0` |
| lowering | `generated_c_cumsum_s8` |
| backend | `cmsis-nn` |

## 网络结构

```text
Input [2,4] -> Q/DQ -> CumSum(axis=1) -> [2,4] -> Q/DQ Output
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
- **数值验收**: `top1=2/2`、最大绝对误差 `0.0`、饱和率 ≤ `0.25`
- **数据集**: `tdd/fixtures/datasets/cumsum_qdq_smoke/dataset.json`

## 边界/风险

axis 以 second input initializer 传入；首个形态 axis=1、exclusive=0、reverse=0。
