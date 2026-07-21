# TRANSPOSE_001: 官方 Transpose QDQ/int8 rank4 显式 perm 数值顺序

## 验证目标

验证 ONNX 官方 `Transpose` 的静态 rank4 QDQ/int8 显式 perm 可以生成真实转置路径并保持输出顺序。

## 来源

内部探索 / 图谱驱动 TDD。`Transpose` 是真实网络导入时最容易暴露 layout 问题的算子。

## ONNX Schema 归属

| 字段 | 内容 |
|------|------|
| domain | `ai.onnx` |
| op_type | `Transpose` |
| opset_range | `11+` |
| schema_form | `static QDQ/int8 tensor, rank=4 explicit perm` |
| lowering | `cmsis_transpose_s8` |
| backend | `cmsis-nn` |

## 网络结构

```text
Input(1,2,2,3) -> Q -> DQ -> Transpose(perm=[0,2,3,1]) -> Q -> DQ -> Output(1,2,3,2)
```

## 输入

- **张量形状**: `[1, 2, 2, 3]`
- **ONNX 边界数据类型**: float32
- **运行期核心数据类型**: int8

## 算子参数

| 算子 | 参数 |
|------|------|
| Transpose | perm=[0,2,3,1] |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|------------|------|
| input/input_q | 0.05 | 0 | float 输入进入 int8 量化域 |
| output/output_q | 0.05 | 0 | 与输入一致，便于精确检查顺序 |

## 预期结果

- **codegen status**: `ok`
- 生成物包含 `arm_transpose_s8`
- ONNX Runtime 与生成 C 输出完全一致

## 数值验收

- **是否需要**: `yes`
- **数据集**: `tdd/fixtures/datasets/transpose_qdq_smoke/dataset.json`
- **通过阈值**: top1 一致率 1.0，最大绝对误差 0.0，饱和率不超过 0.25

## 边界/风险

暂不声明动态 rank、缺省 reverse perm 和跨内部 NHWC runtime 的复杂中间 layout。
