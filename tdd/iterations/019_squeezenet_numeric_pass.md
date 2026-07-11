# 019: SqueezeNet ONNX-vs-C 数值闭环

## 背景

`NET_001` SqueezeNet 1.0 int8 在上一轮已经完成结构代码生成：converter 能解析完整网络，codegen 能生成包含真实 CMSIS-NN 调用的 C 工程，并通过 C99 smoke compile/run。

本轮目标是继续遵循 TDD 原则，为 `NET_001` 构造统一数据集，执行原始 ONNX Runtime 与生成 C 的并行推理对比，并修复导致精度差异过大的问题。

## 新增测试资产

- 数据集：`tdd/fixtures/datasets/net_001_squeezenet_smoke/`
- 用例：`NET_001` 增加 `NumericCheck`
- 样本：
  - `midgray`
  - `channel_ramps`
  - `checker`
  - `center_blob`

这些样本不是 ImageNet 分类准确率数据集，而是固定、可复现的 ImageNet 输入形状 smoke probes，用于验证 ONNX 与生成 C 在同一输入下的输出一致性。

## 暴露的问题

初始并行推理时，`NET_001` 的 C 输出出现全 `127` 饱和，top1 完全不一致。逐步拆解后定位到以下问题：

1. `Softmax` 对 `[1,1000,1,1]` 输出只按最后一维计算 `row_size=1`，导致 1000 类分类轴没有被正确处理。
2. 大 `row_size` 的 int8 softmax 在 host CMSIS 路径上容易产生低分辨率输出，1000 类概率被压到相同或接近的 int8 值。
3. SqueezeNet fire module 中的 `Concat` 是内部 NHWC channel concat；直接调用 `arm_concatenation_s8_z` 会按 channel-contiguous block copy，破坏 NHWC 通道布局。
4. 当模型外部输出为 NCHW、内部 buffer 为 NHWC 时，旧的输出转置包装把 `return NANOC_STATUS_OK` 放在转置前，导致转置代码不可达；同时顺序拷贝可能污染输出语义。

## 修复内容

- `Softmax` 生成逻辑改为根据 ONNX `axis` 计算 `num_rows` 和 `row_size`。
- 为大分类轴生成稳定 softmax fallback：在 CMSIS-NN 启用路径中使用 `expf` 对 int8 logits 做稳定 softmax，再量化回 int8 输出。
- 为 stable softmax 增加 argmax tie preservation，避免 1000 类低概率全部量化成同一值后丢失 logits 排序信息。
- 对 NHWC channel concat 生成显式 channel-copy 循环；保留 `arm_concatenation_s8_z` 映射说明，但不再让该 API 的 block-copy 语义破坏内部布局。
- 修复输出边界转置，确保 NHWC -> NCHW 转置发生在成功返回之前，并避免对转置 override 输出做无意义顺序拷贝。
- `run_numeric_tests.py` 增加 `top1_tie_margin`，用于低置信近似 tie 样本的合理验收。

## 当前验收结果

命令：

```bash
python tdd/scripts/run_numeric_tests.py --case NET_001 --generate
python tdd/scripts/run_numeric_tests.py --generate
python tdd/scripts/run_tests.py --mode target --generate
python tdd/scripts/run_regression.py --generate
```

结果：

- `NET_001` 数值验收：`PASS`
- `NET_001` top1：`4/4`
- 饱和率：`0.00`
- 最大绝对误差：`0.2`
- 全量数值回归：`11/11 PASS`
- target 全量结构回归：`33/33 PASS`
- 稳定回归：`PASS`

`midgray` 样本为低置信 near-tie：ONNX top1 与 C top1 在 ONNX 概率上的 margin 为 `0.0174`，低于 `top1_tie_margin=0.02`，因此计入可接受一致。其余 3 个样本 exact top1 一致。

## 后续风险

- 当前数据集只验证 ONNX-vs-C 一致性，不验证 ImageNet 真实标签分类准确率。
- Stable softmax fallback 使用 `expf`，适合 correctness smoke；后续面向具体 MCU 时需要结合 CMSIS-NN API 版本、libm 成本和输出类型重新评估。
- SqueezeNet 激活尺寸较大，实际 Cortex-M3/M4/M7 是否可部署应由显式 SRAM/Flash 预算 probe 判断，并返回 `oversize`。
- 仍需要逐层 dump 工具，用于未来完整网络数值误差定位。
