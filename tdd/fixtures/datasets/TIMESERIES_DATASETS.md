# Time-Series Dataset Fixtures

本目录下的 `timeseries_*` 数据集用于把已下载的时域/传感器 ONNX 先运行起来，作为后续 ONNX-vs-C 数值回归的入口。

| Dataset | Models | Task | Source kind | Reasonability rule |
|---|---|---|---|---|
| `timeseries_cwru_bearing_smoke` | `timeseries_cwru_bearing_mlp.onnx` | bearing vibration binary classification | real source samples | top1 should match provided healthy/faulty labels for most fixed samples |
| `timeseries_kws_dscnn_smoke` | `timeseries_kws_dscnn_small_int8.onnx`, `timeseries_kws_dscnn_small_qat_int8.onnx` | keyword spotting log-mel classification | deterministic probes | outputs must be finite and stable across reruns; no accuracy claim yet |
| `timeseries_stwin_vowel_smoke` | `timeseries_stwin_vowel.onnx` | IMU vowel gesture classification | deterministic probes | outputs should be finite 5-class distributions |
| `timeseries_met_hybrid_smoke` | `timeseries_met_hybrid.onnx` | MET activity-intensity classification | deterministic probes | outputs should be finite 4-class distributions over increasing motion probes |

说明：除 CWRU 使用原仓库 held-out 样本外，本轮其他数据集主要是固定探针，不宣称真实准确率。
这些探针的价值是确保模型可加载、输入格式明确、输出可复现，并为下一步 C 端数值对比建立样本契约。
