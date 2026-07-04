# QLINEAR_NUM_003: 高通道 QLinearConv per-channel 精度（仿 MNIST Conv1）

## 验证目标

用 MNIST 第一层 QLinearConv 的真实参数构建最小模型（1→8 通道，5×5 kernel，per-channel），验证 C 输出与 ONNX 数值一致。

## 来源

缺陷复现

## 网络结构

```
Float Input(1,1,8,8)
  -> QuantizeLinear(scale=1.0, zp=0, uint8)
  -> QLinearConv(kernel=5x5, out_ch=8, pads=[0,0,0,0], strides=[1,1])
  -> DequantizeLinear(scale=3.68, zp=0, uint8)
  -> Float Output(1,8,4,4)
```

## 输入

- **张量形状**: `[1, 1, 8, 8]`
- **ONNX 边界数据类型**: float32

## 算子参数

| 算子 | 参数 |
|------|------|
| QuantizeLinear | scale=1.0, zp=0 (uint8) |
| QLinearConv | kernel=5×5, out_ch=8, pad=0, stride=1, 8 通道 per-channel w_scale |
| QLinearConv weight | 随机 int8 [8,1,5,5], w_zp=0 |
| QLinearConv bias | int32 [8] 随机值 |
| DequantizeLinear | scale=3.68, zp=0 (uint8) |

## 预期结果

- **codegen status**: `ok`
- CMSIS-NN API: `arm_convolve_wrapper_s8`
- C int8 输出与 ONNX int8 输出差 ≤2（宽松舍入容差，用于 8 通道大 kernel）

## 边界/风险

- 本用例使用随机权重，但量化参数结构与 MNIST Conv1 完全相同
- 8 通道 per-channel multiplier/shift + 5×5 kernel 的组合可能暴露 CMSIS-NN 累积精度问题

## 量化设计

| 参数 | ONNX uint8 | CMSIS-NN int8 |
|------|-----------|---------------|
| 输入 zp | 0 | -128 |
| input_offset | — | 128 |
| 权重 zp | [0]*8 (int8) | [0]*8 |
| 输出 zp | 0 | -128 |
| output_offset | — | -128 |
| multiplier/shift | per-channel | 8 values each |
