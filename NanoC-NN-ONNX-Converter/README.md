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

如需生成样例 ONNX 模型，可额外安装：

```bash
python -m pip install -e ".[examples]"
```

## 计划文档

技术路线和里程碑见 `Doc/plan.md`。
