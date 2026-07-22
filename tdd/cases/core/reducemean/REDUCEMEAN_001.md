# REDUCEMEAN_001: 官方 ReduceMean QDQ/int8 rank2 axes=1 keepdims=1 数值精度

## 验证目标

验证 ONNX 官方 `ReduceMean` 的小型 rank2 行均值形态可生成 C 并通过数值对比。

## 来源

内部探索：人工构造最小官方 ONNX schema 用例。

## ONNX Schema 归属

- `ai.onnx/ReduceMean`
- opset: 11+
- support id: `ONNX_REDUCEMEAN_QDQ_INT8`

## 网络结构

`input[2,4] -> QuantizeLinear -> DequantizeLinear -> ReduceMean(axes=1, keepdims=1) -> QuantizeLinear -> DequantizeLinear -> output[2,1]`

## 输入

- shape: `[2, 4]`
- dtype: float32 API 输入，内部 QDQ/int8

## 算子参数

- axes: `[1]`
- keepdims: `1`

## 量化设计

- input/output scale: `0.05`
- input/output zero_point: `0`

## 预期结果

**codegen status**: `ok`

生成 `generated_c_reducemean_s8` C99 路径，逐行求均值后再量化。

## 数值验收

- dataset: `reducemean_qdq_smoke`
- max_abs_error: `0.06`
- saturation ratio: `<= 0.25`

## 边界/风险

当前只支持 rank2、单 axis、`keepdims=1`。更通用 reduce 需要后续从真实模型继续提炼。
