# CLIP_001: 官方 Clip QDQ/int8 常量 min/max 数值精度

## 验证目标

验证独立 `Clip` 数据路径可生成 C，而不是只能作为 Conv/FC 激活融合存在。

## 来源

内部探索：人工构造最小官方 ONNX schema 用例。

## ONNX Schema 归属

- `ai.onnx/Clip`
- opset: 11+
- support id: `ONNX_CLIP_QDQ_INT8`

## 网络结构

`input -> QuantizeLinear -> DequantizeLinear -> Clip(min,max) -> QuantizeLinear -> DequantizeLinear -> output`

## 输入

- shape: `[1, 8]`
- dtype: float32 API 输入，内部 QDQ/int8

## 算子参数

- min: `-0.10`
- max: `0.20`

## 量化设计

- input/output scale: `0.05`
- input/output zero_point: `0`

## 预期结果

**codegen status**: `ok`

生成 `generated_c_clip_s8` C99 路径，低于 min 的值夹到 min，高于 max 的值夹到 max。

## 数值验收

- dataset: `clip_qdq_smoke`
- max_abs_error: `0.0`
- saturation ratio: `<= 0.25`

## 边界/风险

Standalone Clip 与 activation fusion 是两个语义层。后续需要继续保留融合优化，但不能把独立输出 Clip 误折叠为空。
