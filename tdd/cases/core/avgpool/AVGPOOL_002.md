# AVGPOOL_002: 官方 GlobalAveragePool QDQ/int8 全局池化

## 验证目标

验证 ONNX 官方 `GlobalAveragePool` 在 Q/DQ int8 边界下可以生成真实 CMSIS-NN
`arm_avgpool_s8` 调用，并通过 ONNX-vs-C 数值验收。

## 来源

内部探索 / 真实模型反向提炼。`timeseries_kws_dscnn_small_int8.onnx`、
`timeseries_kws_dscnn_small_qat_int8.onnx`、`timeseries_stwin_vowel.onnx`
均包含官方 `GlobalAveragePool`。本用例先提炼最小静态 rank=4 NCHW 形态，
避免完整 KWS/IMU 网络失败时定位成本过高。

## ONNX Schema 归属

| 字段 | 内容 |
|------|------|
| domain | `ai.onnx` |
| op_type | `GlobalAveragePool` |
| opset_range | `11+` |
| schema_form | `rank=4 static NCHW, QDQ/int8 tensor, full spatial H/W average` |
| lowering | `cmsis_global_avgpool_s8` |
| backend | `cmsis-nn` |

## 网络结构

```text
Input(1,4,3,3)
  -> QuantizeLinear
  -> DequantizeLinear
  -> GlobalAveragePool
  -> QuantizeLinear
  -> DequantizeLinear
  -> Output(1,4,1,1)
```

## 输入

- **张量形状**: `[1, 4, 3, 3]`（NCHW）
- **ONNX 边界数据类型**: float32
- **运行期核心数据类型**: int8

## 算子参数

| 算子 | 参数 |
|------|------|
| GlobalAveragePool | kernel 覆盖完整 H/W，即 3x3 |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|------------|------|
| input/input_q | 0.05 | 0 | float 输入进入 int8 量化域 |
| output/output_q | 0.05 | 0 | 与输入保持一致，先保护 CMSIS-NN avgpool 基本路径 |

## 预期结果

- **codegen status**: `ok`
- 生成代码包含 `arm_avgpool_s8`
- 生成物通过 C99 smoke compile
- ONNX Runtime 与生成 C 输出 top1 一致，最大绝对误差不超过阈值

## 数值验收

- **是否需要**: `yes`
- **数据集**: `tdd/fixtures/datasets/avgpool_global_qdq_smoke/dataset.json`
- **参考路径**: 原始 ONNX Runtime
- **通过阈值**: top1 一致率 1.0，最大绝对误差 0.06，饱和率不超过 0.25

## 边界/风险

- 当前只确认静态 rank=4 NCHW、全局空间池化、输入输出量化参数一致的形态。
- 普通 `AveragePool`、不同输入输出量化参数、NHWC 原生图仍需后续独立 case。
