# REDUCEMIN_001: 官方 ReduceMin QDQ/int8 rank2 axes=1 keepdims=1 数值精度

## 验证目标

验证 ONNX 官方 `ReduceMin` 在 rank2 行归约形态上可生成可执行 C。

## 来源

内部探索：人工构造最小官方 ONNX schema 用例。

## ONNX Schema 归属

- `ai.onnx/ReduceMin`
- opset: 11+
- support id: `ONNX_REDUCEMIN_QDQ_INT8`

## 网络结构

`input[2,4] -> Q/DQ -> ReduceMin(axes=1, keepdims=1) -> Q/DQ -> output[2,1]`

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

生成 `generated_c_reducemin_s8` C99 correctness baseline。

## 数值验收

- dataset: `reducemin_qdq_smoke`
- max_abs_error: `0.0`
- saturation ratio: `<= 0.25`

## 边界/风险

当前只支持 rank2、单 axis、`keepdims=1`。
