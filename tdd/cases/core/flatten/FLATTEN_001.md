# FLATTEN_001: 官方 Flatten QDQ/int8 axis=1 数值顺序

## 验证目标

验证 ONNX 官方 `Flatten` 在 Q/DQ int8 边界下可以保持 ONNX row-major
展开顺序，并在生成 C 推理路径中正确传递到输出。

## 来源

内部探索 / 真实模型反向提炼。`mnist-12-int8.onnx`、`signal_jump.int8.onnx`、
`keyword_spotting_dscnn.int8.onnx` 和多个时序 fixture 都包含 Flatten 或等价
Reshape。MNIST 复盘中也明确指出 layout flatten 是完整网络数值正确性的关键风险点。

## ONNX Schema 归属

| 字段 | 内容 |
|------|------|
| domain | `ai.onnx` |
| op_type | `Flatten` |
| opset_range | `11+` |
| schema_form | `static QDQ/int8 tensor, axis=1, preserves ONNX row-major flatten order` |
| lowering | `generation_time_shape_alias_or_layout_copy` |
| backend | `cmsis-nn` |

## 网络结构

```text
Input(1,2,2,3)
  -> QuantizeLinear
  -> DequantizeLinear
  -> Flatten(axis=1)
  -> QuantizeLinear
  -> DequantizeLinear
  -> Output(1,12)
```

## 输入

- **张量形状**: `[1, 2, 2, 3]`（NCHW）
- **ONNX 边界数据类型**: float32
- **运行期核心数据类型**: int8

## 算子参数

| 算子 | 参数 |
|------|------|
| Flatten | axis=1 |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|------------|------|
| input/input_q | 0.05 | 0 | float 输入进入 int8 量化域 |
| output/output_q | 0.05 | 0 | 与输入一致，便于精确检查顺序 |

## 预期结果

- **codegen status**: `ok`
- 生成物通过 C99 smoke compile
- ONNX Runtime 与生成 C 输出完全一致

## 数值验收

- **是否需要**: `yes`
- **数据集**: `tdd/fixtures/datasets/flatten_axis1_qdq_smoke/dataset.json`
- **参考路径**: 原始 ONNX Runtime
- **通过阈值**: top1 一致率 1.0，最大绝对误差 0.0，饱和率不超过 0.25

## 边界/风险

- 当前只确认静态 shape、axis=1、输入输出量化参数一致的最小形态。
- `Conv/Pool` 内部 NHWC buffer 到 ONNX NCHW flatten 的重排仍由完整拓扑和后续 layout case 继续保护。
