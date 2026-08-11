# REDUCEL1_001: 官方 ReduceL1 QDQ/int8 行归约

## 验证目标

验证 rank=2 QDQ/int8 tensor 在 `axes=[1]`、`keepdims=1` 下执行 L1 归约，并与 ONNX Runtime 数值一致。

## 来源

内部探索：沿已支持的 reduce family 扩展 L1 范数归约能力。

## ONNX Schema 归属

| 字段 | 值 |
|------|----|
| domain | `ai.onnx` |
| op_type | `ReduceL1` |
| opset_range | `11+` |
| schema_form | `rank=2 static QDQ/int8 tensor, axes=1, keepdims=1` |
| lowering | `generated_c_reducel1_s8` |
| backend | `cmsis-nn` |

## 网络结构

```text
Input -> QuantizeLinear -> DequantizeLinear -> ReduceL1 -> QuantizeLinear -> DequantizeLinear -> Output
```

## 输入

- **张量形状**: `[2, 4]`
- **数据类型**: int8 QDQ 数据路径

## 算子参数

| 算子 | 参数 |
|------|------|
| ReduceL1 | `axes=[1]`, `keepdims=1` |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|------------|------|
| input | `0.05` | `0` | 输入量化 |
| output | `0.05` | `0` | 输出 L1 范数量化 |

## 预期结果

- **codegen status**: `ok`
- **若 ok**: 编译通过、运行不崩溃
- **关键验证点**: 生成 C 应执行绝对值求和，而不是普通 sum

## 数值验收

- **是否需要**: `yes`
- **数据集**: `tdd/fixtures/datasets/reducel1_qdq_smoke/dataset.json`
- **参考路径**: 原始 ONNX Runtime
- **通过阈值**: top1 一致率 `1.0`，最大绝对误差 `0.06`，饱和率不超过 `0.25`

## 边界/风险

当前只覆盖 rank=2 行归约，不覆盖多轴、负轴、`keepdims=0` 或空 axes。
