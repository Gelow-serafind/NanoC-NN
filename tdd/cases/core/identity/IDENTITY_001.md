# IDENTITY_001: 官方 Identity QDQ/int8 透传

## 验证目标

验证 `Identity` 对静态 QDQ/int8 tensor 折叠为输入输出透传（memcpy），输出与 ONNX Runtime 数值一致。

## 来源

内部探索：100% 计划 W1c 折叠批；同时修复 empty-runtime 假阳性（此前 report ok 但生成 blocked stub）。

## ONNX Schema 归属

| 字段 | 值 |
|------|----|
| domain | `ai.onnx` |
| op_type | `Identity` |
| opset_range | `1+` |
| schema_form | `static QDQ/int8 tensor, generation-time identity alias` |
| lowering | `generation_time_identity_passthrough` |
| backend | `cmsis-nn` |

## 网络结构

```text
Input -> Q/DQ -> Identity -> Q/DQ Output
```

## 输入

- **张量形状**: `[1, 8]`
- **数据类型**: int8 QDQ 数据路径

## 算子参数

| 算子 | 参数 |
|------|------|
| Identity | 无参数，逐元素透传 |

## 量化设计

| 张量 | scale | zero_point |
|------|-------|------------|
| input | `0.05` | `0` |
| output | `0.05` | `0` |

## 预期结果

- **codegen status**: `ok`
- **数值验收**: `top1=2/2`、最大绝对误差 `0.0`、饱和率 ≤ `0.25`
- **数据集**: `tdd/fixtures/datasets/identity_qdq_smoke/dataset.json`

## 边界/风险

生成 C 以 `memcpy` 透传，无运行期算子。
