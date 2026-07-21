# AVGPOOL_003: 官方 AveragePool QDQ/int8 2x2 stride=2 数值精度

## 验证目标

验证 ONNX 官方 `AveragePool` 在 Q/DQ int8 边界下可以生成 `arm_avgpool_s8`
路径，并与 ONNX Runtime 输出保持一致。

## 来源

内部探索 / 图谱驱动 TDD。当前已有 `GlobalAveragePool`，本用例补齐普通窗口平均池化。

## ONNX Schema 归属

| 字段 | 内容 |
|------|------|
| domain | `ai.onnx` |
| op_type | `AveragePool` |
| opset_range | `11+` |
| schema_form | `rank=4 static NCHW, QDQ/int8 tensor, CMSIS-compatible average pool window` |
| lowering | `cmsis_avgpool_s8` |
| backend | `cmsis-nn` |

## 网络结构

```text
Input(1,2,4,4) -> Q -> DQ -> AveragePool -> Q -> DQ -> Output(1,2,2,2)
```

## 输入

- **张量形状**: `[1, 2, 4, 4]`
- **ONNX 边界数据类型**: float32
- **运行期核心数据类型**: int8

## 算子参数

| 算子 | 参数 |
|------|------|
| AveragePool | kernel_shape=[2,2], strides=[2,2], pads=[0,0,0,0] |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|------------|------|
| input/input_q | 0.05 | 0 | float 输入进入 int8 量化域 |
| output/output_q | 0.05 | 0 | 与输入一致，聚焦池化语义 |

## 预期结果

- **codegen status**: `ok`
- 生成物包含 `arm_avgpool_s8`
- ONNX Runtime 与生成 C 输出误差在阈值内

## 数值验收

- **是否需要**: `yes`
- **数据集**: `tdd/fixtures/datasets/avgpool_window_qdq_smoke/dataset.json`
- **通过阈值**: top1 一致率 1.0，最大绝对误差 0.06，饱和率不超过 0.25

## 边界/风险

暂不声明 `ceil_mode`、`count_include_pad`、非零 padding 和不同输入输出量化。
