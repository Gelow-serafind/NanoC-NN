# QLINEAR_NUM_002: 多通道 QLinearConv uint8 + bias + per-channel

## 验证目标

验证 converter 在多通道 QLinearConv（in_ch=2, out_ch=2, 含 int32 bias, per-channel weight scale）场景下，uint8 量化参数到 CMSIS-NN int8 参数的转换正确性。本用例在 QLINEAR_NUM_001（单通道无 bias）基础上增加复杂度。

## 来源

缺陷复现

## 网络结构

```
Float Input(1,2,2,2)
  -> QuantizeLinear(scale=0.01, zp=128, uint8)
  -> QLinearConv(kernel=1x1, out_ch=2, bias)
  -> DequantizeLinear(scale=0.005, zp=128, uint8)
  -> Float Output(1,2,2,2)
```

## 输入

- **张量形状**: `[1, 2, 2, 2]`（NCHW）
- **ONNX 边界数据类型**: float32

## 算子参数

| 算子 | 参数 |
|------|------|
| QuantizeLinear | scale=0.01, zp=128 (uint8) |
| QLinearConv | kernel_shape=[1,1], strides=[1,1], pads=[0,0,0,0], group=1 |
| QLinearConv weight | int8 shape=[2,2,1,1], w_scale=[0.02, 0.03] per-channel, w_zp=[0,0] |
| QLinearConv bias | int32 [5, -5] |
| DequantizeLinear | scale=0.005, zp=128 (uint8) |

## 量化设计

| 参数 | ONNX uint8 | CMSIS-NN int8 |
|------|-----------|---------------|
| 输入 zp | 128 | 0 |
| input_offset | — | 0 |
| 权重 zp | [0, 0] | [0, 0] |
| filter_offset | — | 0 |
| 输出 zp | 128 | 0 |
| output_offset | — | 0 |
| multiplier | — | per-channel, 由 x_scale×w_scale/y_scale 导出 |
| shift | — | per-channel |

## 预期结果

- **codegen status**: `ok`
- CMSIS-NN API: `arm_convolve_wrapper_s8`
- model.c 中 per-channel multiplier/shift 数组正确生成

## 边界/风险

- per-channel weight scale 使 requantization 从 per-tensor 变为 per-channel，converter 需正确处理 multiplier/shift 数组
- bias 的 int32 值需根据 x_scale×w_scale 正确重新量化
- 当前 converter 可能在此场景下有精度偏差，预期本用例初始 FAIL
