# example-2-detect-signal-jump

本样例训练一个轻量 1D CNN，用于判断长度为 10 的时域信号窗口中是否存在向上或向下突变。

## 方案选择

这个问题的传统规则解法是计算相邻采样点差分：

```text
diff[i] = x[i + 1] - x[i]
```

如果存在 `diff[i] >= 5`，判定为向上突变；如果存在 `diff[i] <= -5`，判定为向下突变；否则判定为无突变。

本样例使用 1D CNN 来学习这个规则。选择 1D CNN 的原因：

- 窗口长度只有 10，突变是局部相邻点变化，1D 卷积正适合提取局部时域特征。
- 相比 LSTM/RNN，1D CNN 更轻量，结构更简单，更适合作为嵌入式推理样例。
- 导出的 ONNX 能覆盖 `Conv`、`Relu`、`Flatten`、`Gemm` 等后续 converter 需要解析的基础算子。

## 任务定义

- 输入：长度为 10 的 `float32` 时域窗口，shape 为 `[N, 1, 10]`。
- 阈值：相邻采样点差值绝对值 `>= 5` 视为突变。
- 输出：三个类别的 logits，shape 为 `[N, 3]`。

类别定义：

- 类别 0：`no_jump`，无突变。
- 类别 1：`up_jump`，存在向上突变。
- 类别 2：`down_jump`，存在向下突变。

示例：

- `5,5,5,5,5,11,11,11,11,11`：`up_jump`。
- `9,9,9,1,1,1,1,1,1,1`：`down_jump`。
- `3,3,4,4,5,5,6,6,7,7`：`no_jump`。

## 目录结构

```text
example-2-detect-signal-jump/
├── README.md
├── data/
│   ├── inference/
│   │   └── input_windows.csv
│   ├── train/
│   │   └── signal_jump_train.csv
│   └── test/
│       └── signal_jump_test.csv
├── scripts/
│   ├── network.py
│   ├── train_and_export.py
│   ├── run_checkpoint.py
│   └── run_onnx.py
└── outputs/
    ├── checkpoints/
    │   └── signal_jump.pt
    └── onnx/
        ├── signal_jump.onnx
        └── signal_jump.int8.onnx
```

`scripts/network.py` 专门展示网络结构。阅读这个文件即可直接看到本样例使用的 1D CNN。

`data/inference/input_windows.csv` 是固定推理测试数据文件。用户可以把要测试的窗口写入该文件，`run_checkpoint.py` 和 `run_onnx.py` 默认都会读取它。

## 训练并导出 ONNX

在仓库根目录执行：

```bash
conda run -n nanoc-onnx-examples python NanoC-NN-ONNX-Examples/example-2-detect-signal-jump/scripts/train_and_export.py
```

脚本会完成：

1. 生成训练集和测试集 CSV。
2. 训练一个简单 1D CNN。
3. 输出训练准确率和测试准确率。
4. 保存 checkpoint 到 `outputs/checkpoints/signal_jump.pt`。
5. 导出 float32 ONNX 文件到 `outputs/onnx/signal_jump.onnx`。
6. 导出 Q/DQ int8 ONNX 文件到 `outputs/onnx/signal_jump.int8.onnx`。

## 使用 PyTorch checkpoint 推理

默认读取 `data/inference/input_windows.csv` 中的所有窗口，并使用
`outputs/onnx/signal_jump.int8.onnx`：

```bash
conda run -n nanoc-onnx-examples python NanoC-NN-ONNX-Examples/example-2-detect-signal-jump/scripts/run_checkpoint.py
```

传入单个目标窗口时使用 `--window`，窗口必须包含 10 个用逗号分隔的数：

```bash
conda run -n nanoc-onnx-examples python NanoC-NN-ONNX-Examples/example-2-detect-signal-jump/scripts/run_checkpoint.py --window 5,5,5,5,5,11,11,11,11,11
```

使用自定义 CSV 文件时使用 `--input-csv`。CSV 必须包含 `x0` 到 `x9` 共 10 列：

```bash
conda run -n nanoc-onnx-examples python NanoC-NN-ONNX-Examples/example-2-detect-signal-jump/scripts/run_checkpoint.py --input-csv NanoC-NN-ONNX-Examples/example-2-detect-signal-jump/data/inference/input_windows.csv
```

## 使用 ONNX 推理

默认读取 `data/inference/input_windows.csv` 中的所有窗口：

```bash
conda run -n nanoc-onnx-examples python NanoC-NN-ONNX-Examples/example-2-detect-signal-jump/scripts/run_onnx.py
```

传入单个目标窗口时使用 `--window`：

```bash
conda run -n nanoc-onnx-examples python NanoC-NN-ONNX-Examples/example-2-detect-signal-jump/scripts/run_onnx.py --window 9,9,9,1,1,1,1,1,1,1
```

使用自定义 CSV 文件时使用 `--input-csv`：

```bash
conda run -n nanoc-onnx-examples python NanoC-NN-ONNX-Examples/example-2-detect-signal-jump/scripts/run_onnx.py --input-csv NanoC-NN-ONNX-Examples/example-2-detect-signal-jump/data/inference/input_windows.csv
```

如果需要对比 float32 ONNX，可显式指定 `--onnx`：

```bash
conda run -n nanoc-onnx-examples python NanoC-NN-ONNX-Examples/example-2-detect-signal-jump/scripts/run_onnx.py --onnx NanoC-NN-ONNX-Examples/example-2-detect-signal-jump/outputs/onnx/signal_jump.onnx --window 9,9,9,1,1,1,1,1,1,1
```

## 链路测试

本样例的 int8 ONNX 用于覆盖 Q/DQ、Conv、Relu、Flatten、Gemm 的解析链路。
当前 codegen 对 Conv 的真实 CMSIS-NN 渲染仍会报告为 blocked，这是预期结果；
它适合作为后续 Conv s8 支持开发的测试输入。

```bash
conda run -n nanoc-onnx-examples python tools/onnx_to_cmsis_pipeline.py \
  --model NanoC-NN-ONNX-Examples/example-2-detect-signal-jump/outputs/onnx/signal_jump.int8.onnx \
  --layout NCHW \
  --target cortex-m4 \
  --sram-budget 128K \
  --flash-budget 256K
```

## 模型结构

```text
Input(1, 10)
  -> Conv1d(1, 8, kernel_size=2)
  -> ReLU
  -> Conv1d(8, 8, kernel_size=2)
  -> ReLU
  -> Flatten
  -> Linear(64, 3)
```

导出的 int8 ONNX 使用 Q/DQ 形式保存量化边界，后续 converter 可据此测试
`QuantizeLinear`、`DequantizeLinear`、`Conv`、`Relu`、`Flatten`、`Gemm`
等算子解析。
