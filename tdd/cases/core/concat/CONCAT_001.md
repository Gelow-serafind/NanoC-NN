# CONCAT_001: rank=4 channel 维 int8 Concat 数值精度

## 验证目标

验证 ONNX 官方 `Concat` 在 rank=4、NCHW、`axis=1` 的 channel 拼接形态下，可以生成真实 CMSIS-NN `arm_concatenation_s8_z` 调用，并且生成 C 的输出与 ONNX Runtime 输出在量化误差范围内一致。

## 来源

内部探索

本用例来自 `NET_001` SqueezeNet 暴露出的 Q/DQ + `Concat` 缺口，是按照 ONNX 1.22.0 官方 schema catalog 和 support matrix 反向提炼出的最小算子用例。

## 网络结构

```text
InputA(1,1,2,2) -> QuantizeLinear -> DequantizeLinear ┐
                                                       ├-> Concat(axis=1) -> QuantizeLinear -> DequantizeLinear -> Output(1,2,2,2)
InputB(1,1,2,2) -> QuantizeLinear -> DequantizeLinear ┘
```

## 输入

- **输入数量**: 2
- **输入 A 形状**: `[1, 1, 2, 2]`
- **输入 B 形状**: `[1, 1, 2, 2]`
- **ONNX 边界数据类型**: float32
- **C 边界数据类型**: int8，两个输入按 `model_graph.json.inputs` 顺序连续打包到 `nanoc_model_run(input, output)` 的 input buffer

## 算子参数

| 算子 | 参数 |
|------|------|
| QuantizeLinear | scale=0.1, zp=0, int8 |
| Concat | axis=1 |
| DequantizeLinear | scale=0.1, zp=0, int8 |

## 量化设计

| 张量 | scale | zero_point | 说明 |
|------|-------|------------|------|
| input_a | 0.1 | 0 | 与输出保持同量化参数 |
| input_b | 0.1 | 0 | 与输出保持同量化参数 |
| concat_out/output | 0.1 | 0 | Concat 只复制 int8 字节，不做 requantize |

## 预期结果

- **codegen status**: `ok`
- CMSIS-NN API: `arm_concatenation_s8_z` 出现在 `model.c`
- 生成物通过 C99 smoke compile
- 数值验收：ONNX Runtime 与生成 C 的输出 `top1=100%`，最大绝对误差不超过 `0.11`

## 边界/风险

- 当前只确认 rank=4 NCHW、channel 维拼接、所有输入输出使用相同量化参数的 Concat 形态。
- 非 channel 维、不同量化参数、多输出图、动态 shape、NHWC 原生输入仍需要后续独立 case。
- 该用例同时保护多输入 `nanoc_model_run()` input buffer 的偏移映射，避免多个 graph input 被错误映射到同一段内存。
