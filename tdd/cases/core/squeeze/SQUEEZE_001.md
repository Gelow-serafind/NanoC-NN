# SQUEEZE_001: 官方 Squeeze QDQ/int8 静态去 1 维数值顺序

## 验证目标

验证 ONNX 官方 `Squeeze` 在静态 QDQ/int8 场景下去掉 size=1 维，并保持线性存储顺序。

## 来源

内部探索 / 图谱驱动 TDD。`Squeeze` 常出现在分类头和模型导出尾部。

## ONNX Schema 归属

| 字段 | 内容 |
|------|------|
| domain | `ai.onnx` |
| op_type | `Squeeze` |
| opset_range | `11+` |
| schema_form | `static QDQ/int8 tensor, explicit axes removing dimensions of size 1` |
| lowering | `generation_time_shape_alias_or_layout_copy` |
| backend | `cmsis-nn` |

## 网络结构

```text
Input(1,1,2,3) -> Q -> DQ -> Squeeze(axes=[1]) -> Q -> DQ -> Output(1,2,3)
```

## 输入

- **张量形状**: `[1, 1, 2, 3]`
- **ONNX 边界数据类型**: float32
- **运行期核心数据类型**: int8

## 算子参数

| 算子 | 参数 |
|------|------|
| Squeeze | axes=[1] |

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
- **数据集**: `tdd/fixtures/datasets/squeeze_qdq_smoke/dataset.json`
- **通过阈值**: top1 一致率 1.0，最大绝对误差 0.0，饱和率不超过 0.25

## 边界/风险

暂不声明 axes 缺省推断、动态 shape、去掉非 1 维等错误输入处理。
