# REDUCELOGSUM_001: 官方 ReduceLogSum QDQ/int8 rank2 axes=1 keepdims=1

## 验证目标

验证 `ReduceLogSum` 对 rank=2、`axes=[1]`、`keepdims=1` 的 QDQ/int8 tensor 生成 C99 行归约路径，输出与 ONNX Runtime 数值一致。

## 来源

内部探索：ONNX 官方图谱归约扩展，100% 计划 W1d-a。

## ONNX Schema 归属

| 字段 | 值 |
|------|----|
| domain | `ai.onnx` |
| op_type | `ReduceLogSum` |
| opset_range | `13+` |
| schema_form | `rank=2 static QDQ/int8 tensor, axes=1, keepdims=1` |
| lowering | `generated_c_reducelogsum_s8` |
| backend | `cmsis-nn` |

## 网络结构

```text
Input [2,4] -> Q/DQ -> ReduceLogSum(axes=[1], keepdims=1) -> [2,1] -> Q/DQ Output
```

## 输入

- **张量形状**: `[2, 4]` -> `[2, 1]`
- **数据类型**: int8 QDQ 数据路径，行和为正值以定义 log

## 算子参数

| 算子 | 参数 |
|------|------|
| ReduceLogSum | `axes=[1]`、`keepdims=1`，逐行 `log(sum(x))` |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|------------|------|
| input | `0.05` | `0` | 外部输入量化 |
| output | `0.05` | `0` | 输出重新量化 |

## 预期结果

- **codegen status**: `ok`
- **若 ok**: 编译通过、运行不崩溃
- **关键验证点**: 生成 C 中存在逐行求和 + `logf` 归约

## 数值验收

- **是否需要**: `yes`
- **数据集**: `tdd/fixtures/datasets/reducelogsum_qdq_smoke/dataset.json`
- **参考路径**: 原始 ONNX Runtime
- **通过阈值**: top1 一致率 `1.0`，最大绝对误差 `0.06`，饱和率不超过 `0.25`

## 边界/风险

行和必须为正，log 才定义；归约输出经量化取整，允许 `max_abs=0.06`（1 个 int8 单位）容忍取整边界。
