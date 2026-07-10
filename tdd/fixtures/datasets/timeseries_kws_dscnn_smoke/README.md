# Keyword Spotting DS-CNN Smoke Dataset



用于两个 nano-kws INT8/QAT INT8 ONNX 的固定 log-mel 探针样本。

这些样本不是 Speech Commands 真实音频，因此不声称准确率，只用于确认模型可运行、输出有限、类别分布稳定。

合理性观察重点：silence/noise 类探针不应产生 NaN/Inf；两个同架构模型应输出可比较但不必完全相同的 top1。



## Files



- `dataset.json`: 固定输入样本。

- `onnx_inference_summary.json`: ONNX Runtime 推理结果和 top1 摘要。
