# CWRU Bearing Vibration Smoke Dataset



用于振动轴承健康/故障二分类模型的固定推理样本。

优先使用 edge-infer 原仓库的 `test_samples.npz`，包含 held-out 特征窗和标签。

合理性观察重点：真实样本的 ONNX top1 应尽量匹配 `healthy/faulty` 标签；若后续 C 端结果偏离该分布，需要回查 Gemm/Flatten 数值链路。



## Files



- `dataset.json`: 固定输入样本。

- `onnx_inference_summary.json`: ONNX Runtime 推理结果和 top1 摘要。
