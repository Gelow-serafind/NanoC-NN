# NEGOP_001: 官方 Neg QDQ/int8 静态张量数值精度

## 验证目标

验证 ONNX 官方 `Neg` 在静态 QDQ/int8 数据路径上可生成可执行 C。

## 来源

内部探索：人工构造最小官方 ONNX schema 用例。

## ONNX Schema 归属

- `ai.onnx/Neg`
- opset: 11+
- support id: `ONNX_NEG_QDQ_INT8`

## 网络结构

`input -> QuantizeLinear -> DequantizeLinear -> Neg -> QuantizeLinear -> DequantizeLinear -> output`

## 输入

- shape: `[1, 8]`
- dtype: float32 API 输入，内部 QDQ/int8

## 算子参数

无属性。

## 量化设计

- input/output scale: `0.05`
- input/output zero_point: `0`

## 预期结果

**codegen status**: `ok`

生成 `generated_c_neg_s8` C99 路径。

## 数值验收

- dataset: `neg_qdq_smoke`
- max_abs_error: `0.0`
- saturation ratio: `<= 0.25`

## 边界/风险

用例名使用 `NEGOP_001`，避免与已有 `NEG_001` 负向模型测试混淆。
