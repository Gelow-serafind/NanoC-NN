# QLINEAR_NUM_001: 最小 QLinearConv uint8 输入数值精度

## 验证目标

验证 converter 将 QLinearConv 的 uint8 量化参数正确转换为 CMSIS-NN int8 参数，使得 C 代码的 int8 推理输出与 ONNX Runtime 的 int8 中间输出数值一致（允许 ±1 的舍入误差）。

本用例直接针对「uint8 zp=0 → int8 zp=-128 → input_offset=128」这类零点偏移问题。

## 来源

缺陷复现

## 网络结构

```
Float Input(1,1,2,2)
  -> QuantizeLinear(scale=1.0, zp=0, uint8)
  -> QLinearConv(kernel=1x1, out_ch=1, no padding, no bias)
  -> DequantizeLinear(scale=0.01, zp=112, uint8)
  -> Float Output(1,1,2,2)
```

## 输入

- **张量形状**: `[1, 1, 2, 2]`（NCHW: batch=1, channels=1, H=2, W=2）
- **ONNX 边界数据类型**: float32

## 算子参数

| 算子 | 参数 |
|------|------|
| QuantizeLinear | scale=1.0, zp=0 (uint8) |
| QLinearConv | kernel_shape=[1,1], strides=[1,1], pads=[0,0,0,0], group=1, no bias |
| QLinearConv weight | int8 `[[[[42]]]]`, w_scale=0.02, w_zp=0 (uint8) |
| DequantizeLinear | scale=0.01, zp=112 (uint8) |

## 量化设计

| 参数 | ONNX uint8 语义 | CMSIS-NN int8 语义 |
|------|----------------|-------------------|
| 输入 zp | 0 (uint8) | -128 (int8) |
| input_offset | — | 128 (= -(-128)) |
| 权重 zp | 0 (uint8) | 0 (int8) |
| filter_offset | — | 0 |
| 输出 zp | 112 (uint8) | -16 (= 112-128) |
| output_offset | — | -16 |

## 预期结果

- **codegen status**: `ok`
- CMSIS-NN API: `arm_convolve_wrapper_s8` 出现在 model.c 中
- converter 输出的 `model_graph.json` 中 `node_quant.conv.cmsis_nn.input_offset` = 128
- converter 输出的 `model_graph.json` 中 `node_quant.conv.cmsis_nn.output_offset` = -16

## 边界/风险

- 当前 converter 的 `_qlinear_conv_quant_info` 在计算 multiplier/shift 时可能未正确处理 uint8→int8 的零点偏移，导致 C 侧输出与 ONNX 不一致
- 该用例目前只验证参数级正确性（input_offset/output_offset），数值级验证需在 runner 中加入 ONNX-vs-C 对比逻辑
- uint8 zp 为 0 时 uint8→int8 转换是正确的（已验证），非零 zp 的情况需要额外用例覆盖
