# NanoC-NN-ONNX-Converter

ONNX 解析与转换子项目。

该模块计划使用 Python 实现，负责解析 ONNX 模型的计算图结构，导出人类可读的网络层清单，并将模型权重转换为 C 语言头文件，供人工编写的 `model.c` 使用。

## 开发环境

建议使用 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## 目录结构

```text
NanoC-NN-ONNX-Converter/
├── Doc/
│   └── plan.md
├── nanoc_onnx_converter/
│   ├── __init__.py
│   ├── __main__.py
│   ├── c_writer.py
│   ├── cli.py
│   ├── exporter.py
│   ├── model.py
│   ├── naming.py
│   ├── parser.py
│   └── shape.py
├── tests/
├── pyproject.toml
└── README.md
```

## 使用方式

在 `NanoC-NN-ONNX-Converter` 目录下执行：

```bash
python -m nanoc_onnx_converter \
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

`weights.h` 只导出 float32 参数权重；`Reshape` shape、axis 等辅助 initializer 会保留在结构报告中，但不会作为 C 权重数组导出。

## 计划文档

技术路线和里程碑见 `Doc/plan.md`。
