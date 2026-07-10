# MET Hybrid Activity Smoke Dataset



用于双输入 MET 活动强度模型的固定推理探针，覆盖静止、轻微、中等、强运动幅值。

当前样本在模型输入空间中构造，尚未绑定 WISDM/MotionSense/UCI-HAR 原始行和 scaler；因此只作为可运行性/趋势观察。

合理性观察重点：活动幅值变化时 4 类输出应保持有限且可复现；如果所有探针完全同类且置信度饱和，后续需要补真实 scaler+数据源。



## Files



- `dataset.json`: 固定输入样本。

- `onnx_inference_summary.json`: ONNX Runtime 推理结果和 top1 摘要。
