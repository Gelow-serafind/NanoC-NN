# TANH_001: 官方 Tanh QDQ/int8 静态张量数值精度

## 验证目标

验证 ONNX 官方 `Tanh` 在静态 QDQ/int8 数据路径上可生成可执行 C，并与 ONNX Runtime 数值接近。

## 来源

内部探索：人工构造最小官方 ONNX schema 用例。

## ONNX Schema 归属

- `ai.onnx/Tanh`
- opset: 11+
- support id: `ONNX_TANH_QDQ_INT8`

## 网络结构

`input -> QuantizeLinear -> DequantizeLinear -> Tanh -> QuantizeLinear -> DequantizeLinear -> output`

## 输入

- shape: `[1, 8]`
- dtype: float32 API 输入，内部 QDQ/int8

## 算子参数

无属性。

## 量化设计

- input scale: `0.05`, zero_point: `0`
- output scale: `1 / 128`, zero_point: `0`

## 预期结果

**codegen status**: `ok`

生成 `generated_c_tanh_s8` C99 路径，输出与 ONNX Runtime 误差不超过数值阈值。

## 数值验收

- dataset: `tanh_qdq_smoke`
- max_abs_error: `0.01`
- saturation ratio: `<= 0.25`

## 边界/风险

当前是 correctness baseline，使用 `tanhf`，后续 MCU 优化可替换为 LUT 或定点近似。
