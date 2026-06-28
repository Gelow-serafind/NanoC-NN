# NanoC-NN-ONNX-Examples

本目录用于存放生成 ONNX 测试模型的独立样例。每个样例都应能独立生成数据、训练模型，并导出一个可供 `NanoC-NN-ONNX-Converter` 解析的 ONNX 文件。

## 环境约束

样例脚本统一使用独立 conda 环境，不使用开发者机器上的已有个人环境。

环境文件：

```text
NanoC-NN-ONNX-Examples/environment.yml
```

固定版本如下：

- Python：`3.11.15`
- PyTorch：`2.2.2`
- ONNX：`1.16.2`
- ONNX Runtime：`1.18.1`
- NumPy：`1.26.4`

首次使用时创建环境：

```bash
conda env create -f NanoC-NN-ONNX-Examples/environment.yml
```

如果环境已存在，可更新环境：

```bash
conda env update -f NanoC-NN-ONNX-Examples/environment.yml --prune
```

运行样例时统一使用：

```bash
conda run -n nanoc-onnx-examples python <script_path>
```

## 命名规范

每个新样例使用如下目录命名：

```text
example-x-xxxx
```

其中：

- `x`：样例编号，从 1 开始递增。
- `xxxx`：用英文短语简要描述样例作用，单词之间使用 `-` 连接。

例如：

- `example-1-is-over-10`：判断输入数值是否大于 10。
- `example-2-detect-signal-jump`：判断 10 点时域窗口是否存在向上或向下突变。
- `example-3-mnist-small-cnn`：简化 MNIST CNN 分类模型。

## 推荐目录结构

每个样例目录建议包含：

```text
example-x-xxxx/
├── README.md
├── data/
│   ├── inference/
│   ├── train/
│   └── test/
├── scripts/
│   ├── network.py
│   ├── train_and_export.py
│   ├── run_checkpoint.py
│   └── run_onnx.py
└── outputs/
    ├── checkpoints/
    └── onnx/
```

说明：

- `data/inference/`：用户放置推理测试数据的固定目录，推理脚本默认读取该目录中的测试输入。
- `data/train/`：训练集或训练集生成结果。
- `data/test/`：测试集或测试集生成结果。
- `scripts/network.py`：模型网络结构定义。网络相关代码必须单独放在这个文件中，让用户能直接看到网络结构。
- `scripts/train_and_export.py`：生成数据、训练模型、保存 checkpoint、导出 ONNX。
- `scripts/run_checkpoint.py`：直接加载训练好的 PyTorch checkpoint 并执行推理。
- `scripts/run_onnx.py`：直接加载导出的 ONNX 文件并执行推理。
- `outputs/checkpoints/`：训练后的模型权重。
- `outputs/onnx/`：导出的 ONNX 文件。

## 运行约定

样例脚本应尽量满足：

- 可以从样例目录或仓库任意目录运行。
- 固定随机种子，保证结果可复现。
- 输出明确的训练精度、测试精度和 ONNX 文件路径。
- 导出的 ONNX 模型使用固定输入 shape，避免动态维度影响后续 C 端转换。
- 每个样例必须同时提供 checkpoint 推理脚本和 ONNX 推理脚本，用于对比原始框架输出和 ONNX Runtime 输出。
- 每个样例的 README 必须针对该任务说明如何运行训练脚本、如何运行 checkpoint 推理脚本、如何运行 ONNX 推理脚本、如何传入目标参数。
- 每个样例必须提供固定推理数据目录 `data/inference/`，推理脚本默认从该目录读取测试数据；如果支持命令行参数覆盖，也必须在 README 中说明。
- 生成物默认写入 `outputs/`，该目录不纳入版本管理；样例脚本必须能重新生成这些文件。
