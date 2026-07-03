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
        ├── is_over_10.onnx
        └── is_over_10.int8.onnx
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
2. 训练一个简单全连接分类器。
3. 输出测试准确率。
4. 导出 float32 ONNX 文件到 `outputs/onnx/is_over_10.onnx`。
5. 导出 Q/DQ int8 ONNX 文件到 `outputs/onnx/is_over_10.int8.onnx`。

## 使用 PyTorch checkpoint 推理

默认读取 `data/inference/input_values.csv` 中的所有数值，并使用
`outputs/onnx/is_over_10.int8.onnx`：

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

如果需要对比 float32 ONNX，可显式指定 `--onnx`：

```bash
conda run -n nanoc-onnx-examples python NanoC-NN-ONNX-Examples/example-1-is-over-10/scripts/run_onnx.py --onnx NanoC-NN-ONNX-Examples/example-1-is-over-10/outputs/onnx/is_over_10.onnx --value 11
```

## 链路测试

本样例的 int8 ONNX 是当前 converter -> codegen -> CMSIS-NN FC 路径的最小闭环样例：

```bash
conda run -n nanoc-onnx-examples python tools/onnx_to_cmsis_pipeline.py \
  --model NanoC-NN-ONNX-Examples/example-1-is-over-10/outputs/onnx/is_over_10.int8.onnx \
  --layout NHWC \
  --target cortex-m4 \
  --sram-budget 64K \
  --flash-budget 128K
```

## 模型结构

```text
Input(1)
  -> Linear(1, 2)
```

导出的 int8 ONNX 使用 Q/DQ 形式保存量化边界，后续 converter 可据此测试
`QuantizeLinear`、`DequantizeLinear`、`Gemm` 和 CMSIS-NN FC 代码生成。
