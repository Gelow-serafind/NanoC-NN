# WHERE_001: 官方 Where QDQ/int8 静态 mask

## 来源

- ONNX 官方 schema: `Where`
- 内部探索：bool condition 与 QDQ/int8 data-path 的最小闭环

## 目标

验证 `Where(condition, x, y)` 在静态 bool mask、同形状 QDQ/int8 数据输入下，可以生成可编译、可运行，并与 ONNX Runtime 数值一致的 C 推理路径。

## 支持范围

- `condition` 为静态 bool initializer
- `x/y/output` 为同形状 QDQ/int8 tensor
- 当前首个形态固定为 `[1,8]`

## 预期

**codegen status**: `ok`

