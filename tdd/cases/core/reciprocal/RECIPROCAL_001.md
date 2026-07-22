# RECIPROCAL_001: 官方 Reciprocal QDQ/int8 非零输入数值精度

## 验证目标

验证 ONNX 官方 `Reciprocal` 在非零静态 QDQ/int8 数据路径上可生成可执行 C。

## 来源

内部探索：人工构造最小官方 ONNX schema 用例。

## ONNX Schema 归属

- `ai.onnx/Reciprocal`
- opset: 11+
- support id: `ONNX_RECIPROCAL_QDQ_INT8`

## 网络结构

`input -> QuantizeLinear -> DequantizeLinear -> Reciprocal -> QuantizeLinear -> DequantizeLinear -> output`

## 输入

- shape: `[1, 8]`
- dtype: float32 API 输入，内部 QDQ/int8
- value domain: 有限且远离 0

## 算子参数

无属性。

## 量化设计

- input scale: `0.05`, zero_point: `0`
- output scale: `0.02`, zero_point: `0`

## 预期结果

**codegen status**: `ok`

生成 `generated_c_reciprocal_s8` C99 路径。

## 数值验收

- dataset: `reciprocal_qdq_smoke`
- max_abs_error: `0.08`
- saturation ratio: `<= 0.25`

## 边界/风险

当前不覆盖接近 0 的输入。生成 C 会做近零保护，防止 MCU 上出现未定义行为。
