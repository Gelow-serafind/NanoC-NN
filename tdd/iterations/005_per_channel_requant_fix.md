# 迭代 005: per-channel requantization 精度修复

## 基本信息

- **日期**: 2026-07-04
- **前置迭代**: 004
- **触发来源**: MNIST C 推理饱和（全输出 -128/127），QLINEAR_NUM_002 单点通过但复杂场景失败

## 目标

修复 QLinearConv 在多通道（≥8）+ 大 kernel（5×5）场景下的 per-channel requantization 精度问题，使 MNIST 模型的 C 推理输出不再饱和，并尽可能逼近 ONNX 输出。

## 排查结论

1. **Conv1 单点仿真**：C 与 ONNX 逐元素一致（手动模拟验证通过）
2. **MaxPool1 不需要 requantization**：Conv1 output_offset=-128 = Conv2 input_offset 对应的 zp，同零點
3. **Conv2 参数**：per-channel multiplier/shift 数值正确
4. **FC 层 input_offset=128**：与 QuantizeLinear zp=0(uint8)→int8 zp=-128 一致

## 未解决

MNIST C 输出仍饱和（全部 -128/127），与 ONNX 丰富值不匹配。怀疑原因：
- 多层级联下 CMSIS-NN 内部 `arm_nn_requantize` 舍入累积偏差
- 或 float-domain MaxPool（Pooling160）在折叠处理后的量化语义偏差

## 新增测试用例

- `QLINEAR_NUM_003`：仿 MNIST Conv1（1→8ch, 5×5 kernel, per-channel）— PASS（编译+运行）
- 该用例与 QLINEAR_NUM_002 类似，数值精度需后续在 runner 中加入 ONNX 对比验证

## 最终结果

- 能力集：17→17 PASS
- NCHW→NHWC 修复：完成（迭代 004）
- MNIST per-channel 饱和：**待独立深入研究**，需要逐层 dump 中间值对比 ONNX
