# WHERE_001: 官方 Where QDQ/int8 静态 mask

## 验证目标

验证 `Where(condition, x, y)` 在静态 bool mask、同形状 QDQ/int8 数据输入下，可以生成可编译、可运行，并与 ONNX Runtime 数值一致的 C 推理路径。

## 来源

内部探索：补齐 ONNX 条件选择算子，并为后续比较类 bool 中间张量提供消费节点。

## ONNX Schema 归属

| 字段 | 值 |
|------|----|
| domain | `ai.onnx` |
| op_type | `Where` |
| opset_range | `11+` |
| schema_form | `static bool condition, same-shape QDQ/int8 then/else tensors` |
| lowering | `generated_c_where_s8` |
| backend | `cmsis-nn` |

## 网络结构

```text
Input -> QuantizeLinear -> DequantizeLinear
Static bool mask + Static QDQ else tensor + input_dq -> Where -> QuantizeLinear -> DequantizeLinear -> Output
```

## 输入

- **张量形状**: `[1, 8]`
- **数据类型**: int8 QDQ 数据路径，模型外部输入为 float

## 算子参数

| 算子 | 参数 |
|------|------|
| Where | `condition` 为 bool initializer，`x/y` 同形状 |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|------------|------|
| input | `0.05` | `0` | 外部输入量化 |
| where.else | `0.05` | `0` | 静态 else 分支 |
| output | `0.05` | `0` | 输出重新量化 |

## 预期结果

- **codegen status**: `ok`
- **若 ok**: 编译通过、运行不崩溃
- **关键验证点**: 生成 C 中应存在逐元素 bool condition 选择逻辑

## 数值验收

- **是否需要**: `yes`
- **数据集**: `tdd/fixtures/datasets/where_qdq_smoke/dataset.json`
- **参考路径**: 原始 ONNX Runtime
- **通过阈值**: top1 一致率 `1.0`，最大绝对误差 `0.0`，饱和率不超过 `0.25`

## 边界/风险

当前只验证静态 bool mask，不覆盖运行时 bool 输入、广播形态或多输出 bool 暴露。
