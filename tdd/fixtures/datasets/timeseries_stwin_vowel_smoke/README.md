# STWIN Vowel IMU Smoke Dataset



用于 6 通道 20x20 IMU vowel 模型的固定归一化输入探针。

源项目真实数据由 DVC 管理，本轮未固化真实 vowel 采集样本；因此这里只验证推理稳定性和输出分布。

合理性观察重点：输出应为 5 类概率分布；不同运动纹理探针应能触发不同或至少稳定的置信度变化。



## Files



- `dataset.json`: 固定输入样本。

- `onnx_inference_summary.json`: ONNX Runtime 推理结果和 top1 摘要。
