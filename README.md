# NanoC-NN

NanoC-NN 是一个轻量级、跨平台的神经网络推理框架定制开发项目，目标是在嵌入式或资源受限环境中摆脱特定厂商工具链绑定，以标准 C99 代码完成神经网络前向推理部署。

项目采用“双组件工具链”思路，将模型解析与底层计算解耦：

- `NanoC-NN-ONNX-Converter`：ONNX 模型解析与转换工具，负责读取模型结构、导出层级清单，并将权重转换为 C 语言头文件。
- `NanoC-NN-C-Operators`：纯 C 神经网络算子库，负责实现 `conv2d`、`relu`、`linear/gemm`、`max_pool`、`softmax` 等基础算子。

整体流程为：先使用 ONNX 转换工具生成模型结构说明和权重头文件，再人工编写 `model.c` 调用 C 算子库拼接网络，最后在 PC 端与 Python/ONNX Runtime 结果进行精度对齐验证。

## 项目目标

- 将至少一个指定 ONNX 模型转换为可运行的 C 推理工程。
- C 端推理结果与 Python/ONNX Runtime 的单元素绝对误差控制在 `1e-4` 以内。
- 核心代码不依赖平台专有头文件，可在支持标准 C99 的编译器下编译。

## 目录结构

```text
NanoC-NN/
├── AGENTS.md
├── Doc/
│   └── 项目章程.md
├── NanoC-NN-ONNX-Converter/
└── NanoC-NN-C-Operators/
```

## 开发约定

- 仓库协作规则见 `AGENTS.md`。
- 文档优先使用中文。
- Python 子项目目标版本为 Python 3.10+。
- C 子项目目标标准为 C99。
- 生成物、缓存、虚拟环境和构建产物不纳入版本管理。
