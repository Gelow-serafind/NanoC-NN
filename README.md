# CMSIS-NN ONNX 端侧代码生成器

本项目原名 NanoC-NN，现定位调整为 **CMSIS-NN ONNX 端侧代码生成器**。

项目目标是读取 ONNX 模型，解析网络结构、权重和量化信息，并生成基于 Arm CMSIS-NN 的 Cortex-M 端侧推理 C 工程。新的核心方向不再是手写完整 `model.c` 或自研全部底层算子，而是复用 CMSIS-NN 已优化的神经网络内核，由本项目负责 ONNX 到 CMSIS-NN 调用代码、静态内存规划、权重整理和验证工程生成。

项目采用“ONNX 前端 + 代码生成器 + 样例验证”的拆分方式：

- `NanoC-NN-ONNX-Converter`：ONNX 前端解析工具，负责读取模型结构、shape、权重和量化相关信息，导出稳定的 `model_graph.json`、结构摘要和权重资料。
- `NanoC-NN-ONNX-Examples`：ONNX 测试模型样例集合，负责生成可供 converter 验证和教学使用的小模型。
- `NanoC-NN-CMSIS-Codegen`：CMSIS-NN 代码生成子项目，负责把 converter 输出转换为 `model.c`、`model.h`、权重文件、静态缓冲区规划和最小可编译工程。

整体流程为：先使用 ONNX 前端生成结构化中间表示，再由 CMSIS-NN codegen 生成 C 推理代码和工程骨架，最后在 PC 端和目标 Arm 工具链中完成编译、运行与 ONNX Runtime 精度对齐验证。

## 项目目标

- 将至少一个指定 ONNX 模型转换为可编译、可运行的 CMSIS-NN C 推理工程。
- 支持固定 shape、NCHW/NHWC 转换边界明确的 CNN 分类模型。
- 优先面向 CMSIS-NN 高价值路径：`Conv`、`DepthwiseConv`、`FullyConnected`、`Pooling`、`Activation`、`Softmax` 及常见逐元素算子。
- 生成代码不在运行期动态分配内存，激活缓冲区、权重、临时 buffer 均由生成阶段静态规划。
- C 端推理结果与 Python/ONNX Runtime 或量化参考实现完成精度对齐。

## 目录结构

```text
NanoC-NN/
├── AGENTS.md
├── Doc/
│   ├── 总体计划.md
│   └── 项目章程.md
├── NanoC-NN-ONNX-Converter/
├── NanoC-NN-ONNX-Examples/
└── NanoC-NN-CMSIS-Codegen/
```

## 开发约定

- 仓库协作规则见 `AGENTS.md`。
- 总体路线见 `Doc/总体计划.md`，项目章程见 `Doc/项目章程.md`。
- 文档优先使用中文。
- Python 子项目目标版本为 Python 3.10+。
- 生成的 C 代码优先兼容 C99，并以 CMSIS-NN/CMSIS-Core 作为目标依赖。
- 生成物、缓存、虚拟环境和构建产物不纳入版本管理。
