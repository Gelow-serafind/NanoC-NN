# NET_001 SqueezeNet Smoke Dataset

这个数据集用于 `NET_001` SqueezeNet 1.0 int8 的 ONNX-vs-C 并行推理数值验收。

它不是 ImageNet 准确率数据集，而是一组固定、可复现的 ImageNet 输入形状 smoke probes：

- `midgray`: 全图中灰度。
- `channel_ramps`: RGB 三通道空间渐变。
- `checker`: 高频棋盘纹理。
- `center_blob`: 中心亮斑。

验收重点是同一批输入在原始 ONNX Runtime 与生成 C 推理之间保持分类输出一致，并观察饱和率与最大绝对误差。

当前 `NET_001` 使用 near-tie 规则保护低置信样本：如果 C top1 与 ONNX top1 不同，但二者在 ONNX 输出上的概率差不超过 `0.02`，视为可接受一致。这个规则用于处理 1000 类 softmax 在 int8 概率步进下的排序抖动，不代表放宽强分类错误。

后续若需要验证真实分类准确率，应替换或追加带标签的 ImageNet 小样本。
