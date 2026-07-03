# nanoc_nn.converter

ONNX 解析与转换模块。

该模块使用 Python 实现，负责解析 ONNX 模型的计算图结构，导出人类可读的网络层清单、机器可读的 `model_graph.json`，并整理权重与后续量化信息，供 `nanoc_nn.codegen` 生成 CMSIS-NN C 推理工程使用。

## 开发环境

建议使用 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## 目录结构

```text
NanoC-NN/
├── docs/
│   └── converter/
│       ├── README.md
│       └── plan.md
├── src/
│   └── nanoc_nn/
│       └── converter/
│           ├── __init__.py
│           ├── __main__.py
│           ├── c_writer.py
│           ├── cli.py
│           ├── exporter.py
│           ├── model.py
│           ├── naming.py
│           ├── parser.py
│           └── shape.py
├── tests/
│   └── converter/
└── pyproject.toml
```

## 使用方式

在仓库根目录下执行：

```bash
python -m nanoc_nn.converter \
  --model path/to/model.onnx \
  --out build/export \
  --prefix nanoc
```

安装 editable 版本后也可以使用命令行入口：

```bash
nanoc-onnx-converter \
  --model path/to/model.onnx \
  --out build/export \
  --prefix nanoc
```

输出文件：

- `README.md`：输出目录说明和推荐阅读顺序。
- `conversion_report.txt`：纯文本转换报告。
- `model_summary.md`：人工走读用的模型结构摘要。
- `model_graph.json`：后续工具复用的机器可读图结构。
- `weights.h`：float32 权重导出的 C99 头文件。

当前结构解析支持的 ONNX 算子：

- `Conv`
- `Relu`
- `Gemm` / `MatMul`
- `MaxPool`
- `Softmax`
- `Flatten`
- `Reshape`
- `Add`
- `Constant`
- `Transpose`
- `Cast`
- `BatchNormalization`
- `GlobalAveragePool`
- `QuantizeLinear` / `DequantizeLinear`

`weights.h` 只导出 float32 参数权重；`Reshape` shape、axis 等辅助 initializer 会保留在结构报告中，但不会作为 C 权重数组导出。

当前已支持第一版 ONNX Q/DQ 量化提取：per-tensor int8/uint8 的 scale、
zero point、量化权重、int32 bias 和 Fully Connected 所需
multiplier/shift 会写入 `model_graph.json.quantization`。该能力先服务
`Gemm(transB=1)` 到 CMSIS-NN `arm_fully_connected_s8()` 的代码生成；Conv、
Add、Pool 等其它量化算子的完整参数重写仍是后续工作。

## 计划文档

技术路线和里程碑见 `docs/converter/plan.md`。
