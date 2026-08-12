# ARGSMIN_001: 官方 ArgMin QDQ/int8 rank2 axis=1 int32 index 输出数值精度

## 验证目标

验证 ONNX 官方 `ArgMin` 在 rank2 行归约形态上可生成可执行 C，
输出 int32 index 索引张量（分类头能力）。

## 来源

内部探索：人工构造最小官方 ONNX schema 用例，承接 NEG_003 拒绝边界转正。

## ONNX Schema 归属

- `ai.onnx/ArgMin`
- opset: 11+
- support id: `ONNX_ARGMIN_QDQ_INT8`

## 网络结构

`input[1,4] -> Q/DQ -> ArgMin(axis=1, keepdims=1) -> output int64[1]`

## 输入

- shape: `[1, 4]`
- dtype: float32 API 输入，内部 QDQ/int8

## 算子参数

- axis: `1`
- keepdims: `1`
- select_last_index: `0`（默认，取首个最小）

## 量化设计

- input scale: `0.05`
- input zero_point: `0`
- output: int64 index，无量化段（外部 ABI 为 int32 index）

## 预期结果

**codegen status**: `ok`

生成 `generated_c_argmin_index` C99 correctness baseline，经 `nanoc_model_run_index`
入口输出 int32 index（新增非 int8 外部 ABI）。

## 数值验收

- dataset: `argmin_qdq_smoke`
- index_compare: `true`（精确整数相等比较，非 float 反量化）
- max_abs_error: `0.0`
- saturation ratio: `0.0`

## 边界/风险

当前只支持 rank2、单 axis（axis=1）、`keepdims=1`、输入 int8 量化的行扫描形态；
int32 index 输出经 `nanoc_model_run_index` 暴露。`select_last_index=1`（取末最小）暂不支持。
