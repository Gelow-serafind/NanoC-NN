# GLOBALLPPOOL_001: 官方 GlobalLpPool QDQ/int8 全局 Lp 池化

## 验证目标

验证 `GlobalLpPool` 对 rank=4 静态 QDQ/int8 tensor 生成 C99 全局 Lp 池化路径，输出与 ONNX Runtime 数值一致。

## 来源

内部探索：ONNX 官方图谱能力扩展，100% 计划推进。

## ONNX Schema 归属

| 字段 | 值 |
|------|----|
| domain | `ai.onnx` |
| op_type | `GlobalLpPool` |
| opset_range | `1+` |
| schema_form | `rank=4 static NCHW, QDQ/int8 tensor, p=2 global Lp pool` |
| lowering | `generated_c_globallppool_s8` |
| backend | `cmsis-nn` |

## 网络结构

```text
Input [1,2,2,2] -> Q/DQ -> GlobalLpPool(p=2) -> [1,2,1,1] -> Q/DQ Output
```

## 输入

- **张量形状**: `[1,2,2,2]` -> `[1,2,1,1]`
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
- **数据集**: `tdd/fixtures/datasets/globallppool_qdq_smoke/dataset.json`

## 边界/风险

生成 C99 逐通道空间 L2 范数路径。
