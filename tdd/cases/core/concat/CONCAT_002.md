# CONCAT_002: SqueezeNet 风格不同量化分支 Concat

## 验证目标

验证 SqueezeNet fire module 中出现的 `DequantizeLinear -> Concat -> QuantizeLinear` 边界，即两个输入分支拥有不同 scale、Concat 输出拥有目标 scale 的形态，可以进入 int8 contract 并生成可编译 C 推理路径。

## 来源

内部探索 / `NET_001` SqueezeNet 反向提炼。

SqueezeNet 的 fire module 会把 expand1x1 与 expand3x3 两个分支在 channel 维拼接。两个分支来自不同 `QLinearConv`，量化 scale 不总是相同；ONNX 图中通过 `DequantizeLinear -> Concat -> QuantizeLinear` 表达这个浮点域拼接。本用例用于把该形态从完整网络中提炼出来，防止只能支持同量化 byte-copy concat。

## ONNX Schema 归属

| 字段 | 值 |
|------|------|
| domain | `ai.onnx` |
| op_type | `Concat` |
| opset_range | `11+` |
| schema_form | rank=4 NCHW, axis=1 channel concat, DQ/Concat/Q with different input quantization |
| lowering | `cmsis_concatenation_s8_z` with pre-concat requantize |
| backend | `cmsis-nn` |

## 网络结构

```text
InputA(1,1,2,2) -> QuantizeLinear(scale=0.1) -> DequantizeLinear ┐
                                                                ├-> Concat(axis=1) -> QuantizeLinear(scale=0.2) -> DequantizeLinear -> Output(1,2,2,2)
InputB(1,1,2,2) -> QuantizeLinear(scale=0.2) -> DequantizeLinear ┘
```

## 输入

- **输入数量**: 2
- **输入 A 形状**: `[1, 1, 2, 2]`
- **输入 B 形状**: `[1, 1, 2, 2]`
- **数据类型**: int8 运行期，float32 ONNX 边界

## 算子参数

| 算子 | 参数 |
|------|------|
| QuantizeLinear A | scale=0.1, zp=0 |
| QuantizeLinear B | scale=0.2, zp=0 |
| Concat | axis=1 |
| QuantizeLinear output | scale=0.2, zp=0 |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|------------|------|
| input_a | 0.1 | 0 | 分支 A 比输出更细 |
| input_b | 0.2 | 0 | 分支 B 与输出一致 |
| concat_out/output | 0.2 | 0 | Concat 后目标量化参数 |

## 预期结果

- **codegen status**: `ok`
- **若 ok**: 编译通过、运行不崩溃
- **关键验证点**: 生成代码包含 `arm_concatenation_s8_z`，且 converter 不因不同输入 scale 阻塞

## 数值验收

- **是否需要**: `yes`
- **数据集**: `tdd/fixtures/datasets/concat_requant_smoke/dataset.json`
- **参考路径**: 原始 ONNX Runtime
- **通过阈值**: top1 一致率 100%，最大绝对误差不超过 0.21，饱和率不超过 0.25

## 边界/风险

- 当前只确认 rank=4 NCHW、axis=1、两个输入分支的 SqueezeNet 风格。
- 更高输入数、其他 axis、非零 zero_point、NHWC 原生图仍需后续 case。
- 本用例要求显式处理分支 requantize，不能把不同 scale 的字节直接拼接后宣称正确。
