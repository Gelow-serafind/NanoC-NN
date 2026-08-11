# GREATER_001: 官方 Greater QDQ/int8 bool 输出驱动 Where

## 验证目标

验证 `Greater` 可以对同形状 QDQ/int8 tensor 生成 bool condition，并作为 `Where` 输入参与后续 int8 数据选择。

## 来源

内部探索：补齐阈值判断和 mask/select 组合中常见的比较分支。

## ONNX Schema 归属

| 字段 | 值 |
|------|----|
| domain | `ai.onnx` |
| op_type | `Greater` |
| opset_range | `11+` |
| schema_form | `same-shape QDQ/int8 tensors, bool output consumed by Where` |
| lowering | `generated_c_greater_bool` |
| backend | `cmsis-nn` |

## 网络结构

```text
Input -> Q/DQ
Static QDQ rhs + input_dq -> Greater -> Where(input_dq, static zero) -> Q/DQ Output
```

## 输入

- **张量形状**: `[1, 8]`
- **数据类型**: int8 QDQ 数据路径，`Greater` 输出 bool 中间张量

## 算子参数

| 算子 | 参数 |
|------|------|
| Greater | 同形状比较 |
| Where | bool condition 选择 input 或 zero |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|------------|------|
| input | `0.05` | `0` | 比较左输入 |
| rhs | `0.05` | `0` | 比较右输入常量 |
| zero | `0.05` | `0` | Where else 分支 |
| output | `0.05` | `0` | 输出重新量化 |

## 预期结果

- **codegen status**: `ok`
- **若 ok**: 编译通过、运行不崩溃
- **关键验证点**: `Greater` 的 bool 输出应能被 `Where` 正确消费

## 数值验收

- **是否需要**: `yes`
- **数据集**: `tdd/fixtures/datasets/greater_where_qdq_smoke/dataset.json`
- **参考路径**: 原始 ONNX Runtime
- **通过阈值**: top1 一致率 `1.0`，最大绝对误差 `0.0`，饱和率不超过 `0.25`

## 边界/风险

当前不覆盖广播比较、运行时 bool 输出或不同量化参数输入。
