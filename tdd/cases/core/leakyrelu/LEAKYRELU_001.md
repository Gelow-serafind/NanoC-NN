# LEAKYRELU_001: 官方 LeakyRelu QDQ/int8 alpha=0.1 数值精度

## 验证目标

验证 ONNX 官方 `LeakyRelu` 在静态 QDQ/int8 数据路径上可生成可执行 C。

## 来源

内部探索：人工构造最小官方 ONNX schema 用例。

## ONNX Schema 归属

- `ai.onnx/LeakyRelu`
- opset: 11+
- support id: `ONNX_LEAKYRELU_QDQ_INT8`

## 网络结构

`input -> QuantizeLinear -> DequantizeLinear -> LeakyRelu -> QuantizeLinear -> DequantizeLinear -> output`

## 输入

- shape: `[1, 8]`
- dtype: float32 API 输入，内部 QDQ/int8

## 算子参数

- `alpha = 0.1`

## 量化设计

- input/output scale: `0.05`
- input/output zero_point: `0`

## 预期结果

**codegen status**: `ok`

生成 `generated_c_leakyrelu_s8` C99 路径，负数分支乘以 `alpha` 后再量化。

## 数值验收

- dataset: `leakyrelu_qdq_smoke`
- max_abs_error: `0.06`
- saturation ratio: `<= 0.25`

## 边界/风险

当前仅保护标量 alpha，后续可增加不同 alpha 和更宽动态范围。
