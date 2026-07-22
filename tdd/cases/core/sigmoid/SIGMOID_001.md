# SIGMOID_001: 官方 Sigmoid QDQ/int8 静态张量数值精度

## 验证目标

验证 ONNX 官方 `Sigmoid` 的 QDQ/int8 静态张量形态可以生成真实 C 推理路径。

## 来源

内部探索 / 图谱驱动 TDD。该用例覆盖 MCU 小模型中常见的逐元素激活函数。

## ONNX Schema 归属

| 字段 | 内容 |
|------|------|
| domain | `ai.onnx` |
| op_type | `Sigmoid` |
| opset_range | `11+` |
| schema_form | `static QDQ/int8 tensor` |
| lowering | `generated_c_sigmoid_s8` |
| backend | `cmsis-nn` |

## 网络结构

```text
Input(1,8) -> Q -> DQ -> Sigmoid -> Q -> DQ -> Output(1,8)
```

## 输入

- **张量形状**: `[1, 8]`
- **ONNX 边界数据类型**: float32
- **运行期核心数据类型**: int8

## 算子参数

| 算子 | 参数 |
|------|------|
| Sigmoid | 无属性 |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|------------|------|
| input/input_q | 0.05 | 0 | 动态输入 |
| output/output_q | 1/256 | -128 | 概率输出量化 |

## 预期结果

- **codegen status**: `ok`
- 生成物包含 `generated_c_sigmoid_s8` 路径或等价逐元素 C 代码
- ONNX Runtime 与生成 C 输出误差在阈值内

## 数值验收

- **是否需要**: `yes`
- **数据集**: `tdd/fixtures/datasets/sigmoid_qdq_smoke/dataset.json`
- **参考路径**: 原始 ONNX Runtime
- **通过阈值**: top1 一致率 1.0，最大绝对误差 0.01，饱和率不超过 0.25

## 边界/风险

首个形态优先保证正确性，可使用 `expf`。后续 MCU 优化可替换为 LUT 或定点近似。
