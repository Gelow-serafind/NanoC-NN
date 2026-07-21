# RESHAPE_001: 官方 Reshape QDQ/int8 静态形状数值顺序

## 验证目标

验证 ONNX 官方 `Reshape` 在静态 QDQ/int8 场景下保持线性存储顺序，并能被生成器正确折叠或复制。

## 来源

内部探索 / 图谱驱动 TDD。`Reshape` 是真实网络中高频 shape-only 算子。

## ONNX Schema 归属

| 字段 | 内容 |
|------|------|
| domain | `ai.onnx` |
| op_type | `Reshape` |
| opset_range | `11+` |
| schema_form | `static QDQ/int8 tensor, constant shape, same element count` |
| lowering | `generation_time_shape_alias_or_layout_copy` |
| backend | `cmsis-nn` |

## 网络结构

```text
Input(1,2,3) -> Q -> DQ -> Reshape(shape=[1,3,2]) -> Q -> DQ -> Output(1,3,2)
```

## 输入

- **张量形状**: `[1, 2, 3]`
- **ONNX 边界数据类型**: float32
- **运行期核心数据类型**: int8

## 算子参数

| 算子 | 参数 |
|------|------|
| Reshape | shape=[1,3,2], allowzero=0 |

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
- **数据集**: `tdd/fixtures/datasets/reshape_qdq_smoke/dataset.json`
- **通过阈值**: top1 一致率 1.0，最大绝对误差 0.0，饱和率不超过 0.25

## 边界/风险

暂不声明动态 shape、`allowzero=1` 和需要改变内存布局的 reshape。
