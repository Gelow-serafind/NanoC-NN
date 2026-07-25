# EQUAL_001: 官方 Equal QDQ/int8 bool 输出驱动 Where

## 来源

- ONNX 官方 schema: `Equal`
- 内部探索：比较类 bool 中间张量

## 目标

验证 `Equal` 可以对同形状 QDQ/int8 tensor 生成 bool condition，并作为 `Where` 的输入参与后续 int8 数据选择。

## 支持范围

- `Equal(input, constant)` 同形状比较
- bool 输出仅作为中间 condition 使用
- 最终模型输出仍为 QDQ/int8，便于 ONNX-vs-C 数值验收

## 预期

**codegen status**: `ok`

