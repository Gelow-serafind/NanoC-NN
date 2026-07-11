# SOFTMAX_002: SqueezeNet 末端 float output Softmax

## 验证目标

验证真实 SqueezeNet 末端 `DequantizeLinear -> Softmax -> float output` 可以由生成器转换为 CMSIS-NN `arm_softmax_s8`，并以 int8 量化概率作为 C 侧输出。

## 来源

内部探索 / `NET_001` SqueezeNet 反向提炼。

SqueezeNet 的 ONNX 输出是 float softmax 概率，但嵌入式 CMSIS-NN 运行接口当前是 `int8_t` input/output。该用例明确本项目的交付策略：末端 float Softmax 在 C 侧以标准 int8 softmax 输出表示，量化参数为 `scale=1/256, zero_point=-128`。

## ONNX Schema 归属

| 字段 | 值 |
|------|------|
| domain | `ai.onnx` |
| op_type | `Softmax` |
| opset_range | `11+` |
| schema_form | quantized logits input, float ONNX output, int8 C softmax output |
| lowering | `cmsis_softmax_s8` |
| backend | `cmsis-nn` |

## 网络结构

```text
Input(1,4)
  -> QuantizeLinear
  -> DequantizeLinear
  -> Softmax(axis=1)
  -> Output(1,4)
```

## 输入

- **张量形状**: `[1, 4]`
- **数据类型**: int8 运行期，float32 ONNX 边界

## 算子参数

| 算子 | 参数 |
|------|------|
| QuantizeLinear | scale=0.1, zp=0 |
| Softmax | axis=1 |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|------------|------|
| input/input_dq | 0.1 | 0 | logits 输入量化 |
| output | 1/256 | -128 | C 侧 int8 softmax 概率输出 |

## 预期结果

- **codegen status**: `ok`
- **若 ok**: 编译通过、运行不崩溃
- **关键验证点**: 生成代码包含 `arm_softmax_s8`，converter 为无输出 Q 的末端 Softmax 推导标准 int8 输出量化参数

## 数值验收

- **是否需要**: `yes`
- **数据集**: `tdd/fixtures/datasets/softmax_float_output_smoke/dataset.json`
- **参考路径**: 原始 ONNX Runtime
- **通过阈值**: top1 一致率 100%，最大绝对误差不超过 0.02，饱和率不超过 0.25

## 边界/风险

- 当前只确认分类向量 Softmax，输出为 C 侧 int8 概率。
- 如果后续需要 C API 直接输出 float，应新增单独 backend/API 设计，而不是改变本用例语义。
