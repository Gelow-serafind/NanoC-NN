# CAST_001: 官方 Cast QDQ/int8 透传（int8→int8）

## 验证目标

验证 `Cast(to=int8)` 对静态 QDQ/int8 tensor 折叠为输入输出透传（memcpy），输出与 ONNX Runtime 数值一致。

## 来源

内部探索：100% 计划 W1c 折叠批；同时修复 empty-runtime 假阳性。

## ONNX Schema 归属

| 字段 | 值 |
|------|----|
| domain | `ai.onnx` |
| op_type | `Cast` |
| opset_range | `1+` |
| schema_form | `static QDQ/int8 tensor, int8-to-int8 generation-time alias` |
| lowering | `generation_time_identity_passthrough` |
| backend | `cmsis-nn` |

## 网络结构

```text
Input -> Q/DQ -> Cast(to=int8) -> Q/DQ Output
```

## 输入

- **张量形状**: `[1, 8]`
- **数据类型**: int8 QDQ 数据路径

## 算子参数

| 算子 | 参数 |
|------|------|
| Cast | `to=INT8`，int8→int8 透传 |

## 量化设计

| 张量 | scale | zero_point |
|------|-------|------------|
| input | `0.05` | `0` |
| output | `0.05` | `0` |

## 预期结果

- **codegen status**: `ok`
- **数值验收**: `top1=2/2`、最大绝对误差 `0.0`、饱和率 ≤ `0.25`
- **数据集**: `tdd/fixtures/datasets/cast_qdq_smoke/dataset.json`

## 边界/风险

仅确认 int8→int8 透传；int8↔float 等跨 dtype Cast 需真实转换路径，未纳入。
