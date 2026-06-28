# nanoc_onnx_converter

这是 `NanoC-NN-ONNX-Converter` 的转换器实现目录，负责把 ONNX 模型解析为人工可读的结构清单、机器可读 JSON，以及 C99 权重头文件。

## 当前目录结构

```text
nanoc_onnx_converter/
├── __init__.py
├── __main__.py
├── cli.py          # 命令行入口，处理参数和错误出口
├── model.py        # 转换过程中的结构化数据模型
├── parser.py       # ONNX 加载、校验、属性解析、节点解析
├── shape.py        # tensor dtype/shape 提取和动态维度处理
├── naming.py       # C 标识符清洗与唯一化
├── c_writer.py     # float32 权重头文件生成
├── exporter.py     # 转换流程编排，生成 Markdown/JSON/C 头文件
└── README.md
```

项目根目录下的 `scripts/convert_onnx.py` 是面向用户的脚本入口，本目录内的 `cli.py` 是包级入口。

## 输出约定

一次转换会在 `--out` 指定目录下生成：

- `README.md`：输出目录说明和推荐阅读顺序。
- `conversion_report.txt`：纯文本转换报告。
- `model_summary.md`：人类可读的模型结构摘要。
- `model_graph.json`：机器可读的结构化图信息。
- `weights.h`：float32 initializer 导出的 C99 头文件。

当前实现保持初期边界：不生成完整 `model.c`，不做 layout transpose，不处理量化权重，不为动态非 batch 维度规划 C 缓冲区。
