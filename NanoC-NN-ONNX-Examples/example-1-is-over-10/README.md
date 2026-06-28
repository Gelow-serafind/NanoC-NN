# example-1-is-over-10

本样例训练一个极小的神经网络，用于判断输入的单个数字是否大于 10。

## 任务定义

- 输入：一个 `float32` 数值，shape 为 `[N, 1]`。
- 输出：两个类别的 logits，shape 为 `[N, 2]`。
- 类别 0：输入值小于或等于 10。
- 类别 1：输入值大于 10。

## 目录结构

```text
example-1-is-over-10/
├── README.md
├── data/
│   ├── inference/
│   │   └── input_values.csv
│   ├── train/
│   │   └── is_over_10_train.csv
│   └── test/
│       └── is_over_10_test.csv
├── scripts/
│   ├── network.py
│   ├── train_and_export.py
│   ├── run_checkpoint.py
│   └── run_onnx.py
└── outputs/
    ├── checkpoints/
    │   └── is_over_10.pt
    └── onnx/
        └── is_over_10.onnx
```

`data/` 和 `outputs/` 中的文件由脚本生成。

其中 `scripts/network.py` 专门用于展示网络结构。阅读这个文件即可直接看到本样例使用的神经网络。

`data/inference/input_values.csv` 是固定推理测试数据文件。用户可以把要测试的数字写入该文件，`run_checkpoint.py` 和 `run_onnx.py` 默认都会读取它。

## 训练并导出 ONNX

在仓库根目录执行：

```bash
conda run -n nanoc-onnx-examples python NanoC-NN-ONNX-Examples/example-1-is-over-10/scripts/train_and_export.py
```

脚本会完成：

1. 生成训练集和测试集 CSV。
2. 训练一个简单 MLP。
3. 输出测试准确率。
4. 导出 ONNX 文件到 `outputs/onnx/is_over_10.onnx`。

## 使用 PyTorch checkpoint 推理

默认读取 `data/inference/input_values.csv` 中的所有数值：

```bash
conda run -n nanoc-onnx-examples python NanoC-NN-ONNX-Examples/example-1-is-over-10/scripts/run_checkpoint.py
```

传入单个目标参数时使用 `--value`：

```bash
conda run -n nanoc-onnx-examples python NanoC-NN-ONNX-Examples/example-1-is-over-10/scripts/run_checkpoint.py --value 11
```

使用自定义 CSV 文件时使用 `--input-csv`。CSV 必须包含名为 `value` 的列：

```bash
conda run -n nanoc-onnx-examples python NanoC-NN-ONNX-Examples/example-1-is-over-10/scripts/run_checkpoint.py --input-csv NanoC-NN-ONNX-Examples/example-1-is-over-10/data/inference/input_values.csv
```

## 使用 ONNX 推理

默认读取 `data/inference/input_values.csv` 中的所有数值：

```bash
conda run -n nanoc-onnx-examples python NanoC-NN-ONNX-Examples/example-1-is-over-10/scripts/run_onnx.py
```

传入单个目标参数时使用 `--value`：

```bash
conda run -n nanoc-onnx-examples python NanoC-NN-ONNX-Examples/example-1-is-over-10/scripts/run_onnx.py --value 11
```

使用自定义 CSV 文件时使用 `--input-csv`：

```bash
conda run -n nanoc-onnx-examples python NanoC-NN-ONNX-Examples/example-1-is-over-10/scripts/run_onnx.py --input-csv NanoC-NN-ONNX-Examples/example-1-is-over-10/data/inference/input_values.csv
```

## 模型结构

```text
Input(1)
  -> Linear(1, 8)
  -> ReLU
  -> Linear(8, 2)
```

导出的 ONNX 模型使用固定 batch size `1` 的示例输入，后续 converter 可据此测试 `Gemm`、`Relu` 等基础算子解析。
