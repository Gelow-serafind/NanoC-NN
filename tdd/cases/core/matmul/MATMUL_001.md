# MATMUL_001: 官方 MatMul QDQ/int8 FC-compatible 形态

## 验证目标

验证 ONNX 官方 `MatMul` 的静态全连接兼容形态可以生成真实 CMSIS-NN
`arm_fully_connected_s8` 调用，并通过 ONNX-vs-C 数值验收。

## 来源

内部探索 / 图谱驱动 TDD。当前完整网络已经在 `QLinearMatMul` 和 `Gemm` 路径上具备 FC 能力，
但官方 `MatMul` 在 support matrix 中仍处于 planned/blocked。该用例用于把
`A[1,I] x B[I,O] -> Y[1,O]` 的常见分类头形态独立沉淀为能力。

## ONNX Schema 归属

| 字段 | 内容 |
|------|------|
| domain | `ai.onnx` |
| op_type | `MatMul` |
| opset_range | `11+` |
| schema_form | `static fully connected compatible form, QDQ/int8 parameters` |
| lowering | `cmsis_fully_connected_s8` |
| backend | `cmsis-nn` |

## 网络结构

```text
Input(1,4)
  -> QuantizeLinear
  -> DequantizeLinear
  -> MatMul(weight[4,3])
  -> QuantizeLinear
  -> DequantizeLinear
  -> Output(1,3)
```

## 输入

- **张量形状**: `[1, 4]`
- **ONNX 边界数据类型**: float32
- **运行期核心数据类型**: int8

## 算子参数

| 算子 | 参数 |
|------|------|
| MatMul | A `[1,4]`，B `[4,3]` |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|------------|------|
| input/input_q | 0.05 | 0 | float 输入进入 int8 量化域 |
| weight | 0.04 | 0 | 常量 int8 权重，converter 需转成 CMSIS FC `[O,I]` |
| output/output_q | 0.04 | 0 | 输出 int8 域 |

## 预期结果

- **codegen status**: `ok`
- 生成代码包含 `arm_fully_connected_s8`
- 生成物通过 C99 smoke compile
- ONNX Runtime 与生成 C 输出 top1 一致，最大绝对误差不超过阈值

## 数值验收

- **是否需要**: `yes`
- **数据集**: `tdd/fixtures/datasets/matmul_qdq_smoke/dataset.json`
- **参考路径**: 原始 ONNX Runtime
- **通过阈值**: top1 一致率 1.0，最大绝对误差 0.08，饱和率不超过 0.25

## 边界/风险

- 当前只确认左输入为单 batch 2-D activation、右输入为静态 rank=2 常量权重的 FC-compatible 形态。
- batched MatMul、动态 shape、双 runtime 输入矩阵乘和高维 broadcast MatMul 仍需后续独立 case。
