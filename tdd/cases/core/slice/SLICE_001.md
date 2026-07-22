# SLICE_001: 官方 Slice QDQ/int8 静态区间数值顺序

## 验证目标

验证 ONNX 官方 `Slice` 的数据路径静态 QDQ/int8 形态可以生成真实 C copy 推理路径。

## 来源

内部探索 / 图谱驱动 TDD。此前 `Slice` 只作为 shape-helper 折叠，本用例将真实数据切片纳入能力集。

## ONNX Schema 归属

| 字段 | 内容 |
|------|------|
| domain | `ai.onnx` |
| op_type | `Slice` |
| opset_range | `11+` |
| schema_form | `static QDQ/int8 tensor, constant starts/ends/axes/steps` |
| lowering | `generated_c_slice_s8` |
| backend | `cmsis-nn` |

## 网络结构

```text
Input(1,1,3,4) -> Q -> DQ -> Slice -> Q -> DQ -> Output(1,1,2,2)
```

## 输入

- **张量形状**: `[1, 1, 3, 4]`
- **ONNX 边界数据类型**: float32
- **运行期核心数据类型**: int8

## 算子参数

| 算子 | 参数 |
|------|------|
| Slice | `starts=[0,0,1,1]`, `ends=[1,1,3,3]`, `axes=[0,1,2,3]`, `steps=[1,1,1,1]` |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|------------|------|
| input/input_q | 0.05 | 0 | 动态输入 |
| output/output_q | 0.05 | 0 | 与输入同量化 |

## 预期结果

- **codegen status**: `ok`
- 生成物包含 `generated_c_slice_s8` 路径或等价 copy C 代码
- ONNX Runtime 与生成 C 输出逐元素一致

## 数值验收

- **是否需要**: `yes`
- **数据集**: `tdd/fixtures/datasets/slice_qdq_smoke/dataset.json`
- **参考路径**: 原始 ONNX Runtime
- **通过阈值**: top1 一致率 1.0，最大绝对误差 0.0，饱和率不超过 0.25

## 边界/风险

shape-helper `Slice` 仍允许折叠；本用例只声明数据路径的静态正步长形态。
