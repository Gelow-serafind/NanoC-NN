# REDUCEL1_001: 官方 ReduceL1 QDQ/int8 行归约

## 来源

- ONNX 官方 schema: `ReduceL1`
- 内部探索：reduce family 扩展

## 目标

验证 rank=2 QDQ/int8 tensor 在 `axes=[1]`、`keepdims=1` 下执行 L1 归约，并与 ONNX Runtime 数值一致。

## 支持范围

- 输入形状 `[2,4]`
- `axes=[1]`
- `keepdims=1`
- 输出 QDQ/int8

## 预期

**codegen status**: `ok`

