# LPPOOL_001: 官方 LpPool QDQ/int8 窗口 Lp 池化

## 验证目标

验证 `LpPool` 对 rank=4 静态 QDQ/int8 tensor 生成 C99 窗口 Lp 池化路径，输出与 ONNX Runtime 数值一致。

## 来源

内部探索：ONNX 官方图谱能力扩展，100% 计划推进。

## ONNX Schema 归属

| 字段 | 值 |
|------|----|
| domain | `ai.onnx` |
| op_type | `LpPool` |
| opset_range | `1+` |
| schema_form | `rank=4 static NCHW, QDQ/int8 tensor, p=2 kernel=2 stride=1 window Lp pool` |
| lowering | `generated_c_lppool_s8` |
| backend | `cmsis-nn` |

## 网络结构

```text
Input [1,1,2,2] -> Q/DQ -> LpPool(p=2, kernel=[2,2], stride=[1,1]) -> [1,1,1,1] -> Q/DQ Output
```

## 输入

- **张量形状**: `[1,1,2,2]` -> `[1,1,1,1]`
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
- **数据集**: `tdd/fixtures/datasets/lppool_qdq_smoke/dataset.json`

## 边界/风险

生成 C99 窗口滑动 L2 范数路径；首个形态固定 kernel=2 stride=1 无 padding。
