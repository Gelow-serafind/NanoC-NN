# 013 Time-Series ONNX Dataset Probes

## 背景

本轮目标是把 5 个时域/传感器类 ONNX 候选先运行起来，并沉淀一批后续可复用的数据集样本。这里的重点不是立即扩大 codegen 能力集，而是建立真实网络数值测试入口：先能稳定执行 ONNX 推理，再逐步升级为 `ONNX Runtime` 与生成 C 代码的结果对比。

## 已固化资产

- `tdd/fixtures/onnx/timeseries_cwru_bearing_mlp.onnx`
- `tdd/fixtures/onnx/timeseries_kws_dscnn_small_int8.onnx`
- `tdd/fixtures/onnx/timeseries_kws_dscnn_small_qat_int8.onnx`
- `tdd/fixtures/onnx/timeseries_met_hybrid.onnx`
- `tdd/fixtures/onnx/timeseries_stwin_vowel.onnx`

数据集与 ONNX 推理结果已固化在：

- `tdd/fixtures/datasets/timeseries_cwru_bearing_smoke/`
- `tdd/fixtures/datasets/timeseries_kws_dscnn_smoke/`
- `tdd/fixtures/datasets/timeseries_stwin_vowel_smoke/`
- `tdd/fixtures/datasets/timeseries_met_hybrid_smoke/`

复现脚本：

```bash
python tdd/scripts/build_timeseries_datasets.py
```

## ONNX 推理结果

### CWRU Bearing MLP

来源为 edge-infer 原仓库的 held-out `test_samples.npz`，因此这一组是本轮最接近真实数值回归的数据集。

| Sample | Expected | ONNX top1 | Result |
|---|---|---|---|
| `cwru_real_00_healthy` | healthy | healthy | PASS |
| `cwru_real_01_healthy` | healthy | healthy | PASS |
| `cwru_real_02_faulty` | faulty | faulty | PASS |
| `cwru_real_03_faulty` | faulty | faulty | PASS |
| `cwru_real_04_faulty` | faulty | faulty | PASS |

结论：5/5 与标签一致，优先升级为下一轮 `ONNX vs generated C` 数值回归。

### KWS DS-CNN INT8 / QAT INT8

本轮使用固定 log-mel 探针，不是 Speech Commands 真实音频，因此不声明准确率。

- `timeseries_kws_dscnn_small_int8.onnx`：5 个探针 top1 均为 `_silence_`，概率约 0.40 到 0.92。
- `timeseries_kws_dscnn_small_qat_int8.onnx`：5 个探针 top1 均为 `_silence_`，概率约 0.999。

结论：两个 ONNX 都可运行且输出有限。当前探针更适合保护输入形状、Q/DQ Conv 链路和输出稳定性；后续需要补 Speech Commands 小样本或真实 log-mel clip 才能做准确率判断。

### STWIN Vowel CNN

本轮使用 6 通道 20x20 归一化 IMU 图像探针，不是真实 vowel 采集样本。

| Sample | ONNX top1 |
|---|---|
| `stwin_neutral_midpoint` | O |
| `stwin_axis0_horizontal_ramp` | A |
| `stwin_axis3_sine_motion` | O |
| `stwin_six_axis_checker` | U |
| `stwin_center_impulse` | I |

结论：模型可运行，且不同探针触发了不同类别，适合作为布局/Conv/Pool/FC smoke；后续若能获取 DVC 数据，应升级为真实标签数值集。

### MET Hybrid Activity

本轮已保留原仓库 scaler sidecar，但使用的样本仍是模型输入空间中的固定运动幅值探针，不是真实 WISDM/MotionSense/UCI-HAR 行。

| Sample | ONNX top1 |
|---|---|
| `met_sedentary_baseline` | light |
| `met_light_periodic_motion` | light |
| `met_moderate_periodic_motion` | light |
| `met_vigorous_periodic_motion` | light |

结论：ONNX 可运行，多输入链路可被执行；但当前探针都偏向 `light`，说明仅有 scaler、没有真实原始窗口和特征行时不能用于准确率判断。它更适合作为后续多输入、MatMul/Conv/Concat/Softmax 结构探索入口。

## 后续建议

1. 优先把 CWRU Bearing MLP 升级为正式数值回归：执行 ONNX 推理、生成 C 代码、编译运行 C runner，并比较二分类输出。
2. KWS DS-CNN 需要获取或构造小型真实 Speech Commands log-mel 数据，再把 `_silence_`、关键词、unknown 三类都覆盖到。
3. STWIN Vowel 需要获取 DVC 管理的真实 IMU 样本，当前探针只能保护推理稳定性。
4. MET Hybrid 已保留 scaler，下一步需要补真实数据预处理和少量真实窗口，否则模型输入空间探针只能用于 smoke，不能用于合理性/准确率验收。
5. 固化下一步测试时要区分 `accuracy dataset` 与 `smoke probe dataset`，禁止把无真实标签的探针写成能力证明。
