# UNSQUEEZE_001: 官方 Unsqueeze QDQ/int8 数据路径静态升维数值顺序

## 验证目标

验证 `Unsqueeze` 在数据路径上不会被误当成 shape helper 折叠，而是能保持元素顺序生成可执行 C。

## 来源

内部探索：人工构造最小官方 ONNX schema 用例。

## ONNX Schema 归属

- `ai.onnx/Unsqueeze`
- opset: 13+
- support id: `ONNX_UNSQUEEZE_QDQ_INT8`

## 网络结构

`input[1,4] -> QuantizeLinear -> DequantizeLinear -> Unsqueeze(axis=1) -> QuantizeLinear -> DequantizeLinear -> output[1,1,4]`

## 输入

- shape: `[1, 4]`
- dtype: float32 API 输入，内部 QDQ/int8

## 算子参数

- axes: `[1]`

## 量化设计

- input/output scale: `0.05`
- input/output zero_point: `0`

## 预期结果

**codegen status**: `ok`

生成 `generated_c_unsqueeze_s8` copy 路径，元素数量不变、线性顺序不变。

## 数值验收

- dataset: `unsqueeze_qdq_smoke`
- max_abs_error: `0.0`
- saturation ratio: `<= 0.25`

## 边界/风险

shape-helper Unsqueeze 仍应在生成期折叠；本用例保护的是带 QDQ 的真实数据路径。
