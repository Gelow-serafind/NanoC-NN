# Time-Series ONNX Candidates

本文件记录 2026-07-08 下载的时域/传感器类小型边缘端 ONNX 候选。它们尚未注册为正式 TDD case，下一步需要逐个提升为 `NET_xxx` 并执行 pipeline 分类。

| 文件 | 场景 | 来源 | SHA256 | 初步结构 |
|------|------|------|--------|----------|
| `timeseries_met_hybrid.onnx` | wearable / IMU 活动强度 MET 估计 | GitHub `acd17sk/MET-Metabolic-Equivalent-of-Task-AI-Android-APP` | `f7519c25adfeed8ab00718128e7d59e685816ecd58c15c61625670605b524e50` | Conv/MatMul/Add/GlobalAveragePool/Softmax，双输入 |
| `timeseries_stwin_vowel.onnx` | STWIN 传感器/audio vowel recognition | GitHub `daleonpz/stwin_AI_vowel_recognition` | `61ed21ce8637d4e6740d6a69f471f7624aa7fe697f23aadb2201b6cb57234f28` | Conv/Relu/MaxPool/GlobalAveragePool/Gemm/Softmax |
| `timeseries_cwru_bearing_mlp.onnx` | CWRU bearing vibration anomaly | GitHub `idan-ben-ami/edge-infer` | `4cafc6aced05e7356dc44e4af1bfa378a0d092fcf2f064ac6ba0d14a20e89577` | Flatten/Gemm/Relu MLP |
| `timeseries_kws_dscnn_small_int8.onnx` | keyword spotting DS-CNN int8 | GitHub `joshleh/nano-kws` | `2c0b7149b9e005badac43eeadf8d1080a66882624f9fd4c92fae2b2ca961ff6b` | Q/DQ int8 Conv stack + GlobalAveragePool + Gemm |
| `timeseries_kws_dscnn_small_qat_int8.onnx` | keyword spotting DS-CNN QAT int8 | GitHub `joshleh/nano-kws` | `04a62e9d6a8310c577ab44b264efe0269aad35588c04328173f277c3b05ba5d1` | Q/DQ int8 Conv stack + GlobalAveragePool + Gemm |

## 初步筛选结论

- 优先建议从 `timeseries_kws_dscnn_small_int8.onnx` 开始，因为它是 int8、DS-CNN、KWS，是 Cortex-M/TinyML 最典型的时域/音频模型之一。
- `timeseries_stwin_vowel.onnx` 体积极小，结构接近当前已支持的 Conv/Pool/FC，适合作为第二个快速 smoke。
- `timeseries_cwru_bearing_mlp.onnx` 是振动异常小 MLP，适合作为工业时域最小网络。
- `timeseries_met_hybrid.onnx` 覆盖多输入、Conv/MatMul/Concat/Softmax，适合暴露多输入和 shape/layout 缺口。
- 两个 KWS 文件架构相同但训练/量化来源不同；正式 case 可以先选一个，另一个作为交叉验证或后续扩展。

## 搜索中剔除的候选

以下文件曾下载到 `tdd/work/model_search/rejected/`，但不作为本轮神经网络候选：

- CalmSense stress model：ONNX 有效，但主体是 `TreeEnsembleClassifier`，不是神经网络。
- conveyor/pump industrial models：ONNX 有效，但主体是 `TreeEnsembleClassifier` 或 `Scaler + TreeEnsembleClassifier`。
- Python_WakeWordDetection 示例模型：下载文件不是可直接解析的 ONNX 模型。
