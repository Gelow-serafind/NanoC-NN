# 迭代 004: QLinearConv 数值精度修复

## 基本信息

- **日期**: 2026-07-04
- **前置迭代**: 003
- **触发来源**: QLINEAR_NUM_002 数值偏差 + MNIST C 推理饱和

## 目标

修复 QLinearConv 在 CMSIS-NN 代码生成中的数值精度问题，使 C 推理输出与 ONNX 一致。

## 新增/修改的测试用例

- `QLINEAR_NUM_001`：单通道无 bias QLinearConv（已有，PASS）
- `QLINEAR_NUM_002`：多通道 + bias + per-channel scale QLinearConv（已有，数值 FAIL）

## 根因

**确认：NCHW→NHWC 数据布局转换缺失。**

- ONNX 和 converter 使用 NCHW 张量布局
- CMSIS-NN `arm_convolve_wrapper_s8` 期望 NHWC 布局
- codegen 的 `_activation_dims` 正确将 shape 描述符从 NCHW 转为 NHWC（如 `[1,C,H,W]→[1,H,W,C]`），但**未对内存中的数据做实际转置**
- 对于 C=1 的单通道模型（如 CONV_001），NCHW 和 NHWC 内存布局完全相同，因此之前的 TDD 测试未暴露该问题

## 验证

修复前（NCHW 输入直接传入 CMSIS-NN）：
```
C: 3 1 8 6 13 11 15 15   (完全错误)
```

修复后（NCHW→NHWC 转置 + CMSIS-NN + NHWC→NCHW 转置）：
```
C: 6 -3 8 0 10 3 12 7    (匹配 ONNX [6,-3,8,0,10,3,12,6]，仅末位±1舍入差)
```

## 代码修改

- `src/nanoc_nn/codegen/generator.py`:
  - 新增 `_cmsis_tensor_runtime_run_body_with_transpose()` — 在 CMSIS-NN 调用前后插入 NCHW↔NHWC 边界转置
  - 新增 `_graph_input_shape()` — 获取模型输入形状
  - 新增 `_tensor_shape()` — 查找张量 NCHW 形状
  - `_model_c()`: 多通道 4D 输入时声明 `nanoc_input_nhwc` 静态缓冲区
  - `_model_c()`: `#else` 块中添加 `(void)nanoc_input_nhwc` 抑制未使用警告

## 最终结果

- QLINEAR_NUM_001: PASS（无变化，单通道不受影响）
- QLINEAR_NUM_002: PASS（修复后 C 输出匹配 ONNX）
- 全量回归: **16/16 PASS**，无能力退化
- 能力集: 16→16，新增 QLINEAR_NUM_001 + QLINEAR_NUM_002
