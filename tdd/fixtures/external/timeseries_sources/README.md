# Time-Series External Source Assets

本目录保存时域/传感器 ONNX 候选附带的小型 sidecar 资产，用于构建后续测试数据集。

## Assets

- `edge_infer/cwru_test_samples.npz`
  - 来源：`idan-ben-ami/edge-infer`
  - 用途：CWRU bearing MLP 的 held-out test samples，包含 `X_test`、`y_test`、`mean`、`std`。
- `nano_kws/*.label_map.json`
  - 来源：`joshleh/nano-kws`
  - 用途：KWS DS-CNN INT8 / QAT INT8 的类别名、输入名、输出名和输入形状。
- `met_app/*.npy`
  - 来源：`acd17sk/MET-Metabolic-Equivalent-of-Task-AI-Android-APP`
  - 用途：MET hybrid 模型的 raw/feature scaler 参数。当前尚未固化真实 WISDM/MotionSense/UCI-HAR 行，因此本轮 MET 数据集仍是 smoke probe，不是准确率集。
