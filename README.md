# CMSIS-NN ONNX 端侧代码生成器

本项目原名 NanoC-NN，现定位调整为 **CMSIS-NN ONNX 端侧代码生成器**。

项目目标是读取 ONNX 模型，解析网络结构、权重和量化信息，并生成基于 Arm CMSIS-NN 的 Cortex-M 端侧推理 C 工程。新的核心方向不再是手写完整 `model.c` 或自研全部底层算子，而是复用 CMSIS-NN 已优化的神经网络内核，由本项目负责 ONNX 到 CMSIS-NN 调用代码、静态内存规划、权重整理和验证工程生成。

本项目的限定场景是 **Arm 架构的嵌入式低算力平台**，典型目标包括 STM32、GD32 等 Cortex-M MCU。工具链优先服务裸机或 RTOS 环境下的小型神经网络部署，关注有限 SRAM/Flash、无动态内存、固定输入尺寸、int8 量化和可交叉编译工程，而不是 Linux、桌面端或 Arm A-class 高性能推理。

项目采用“标准 Python 工程 + ONNX 前端 + CMSIS-NN 代码生成 + 样例验证”的拆分方式：

- `pyproject.toml` 与 `src/nanoc_nn/`：仓库最高层级的 Python 工程入口，提供统一 CLI 和 ONNX 到 CMSIS-NN 的 pipeline 编排。
- `src/nanoc_nn/converter`：ONNX 前端解析模块，负责读取模型结构、shape、权重和量化相关信息，导出稳定的 `model_graph.json`、结构摘要和权重资料。
- `src/nanoc_nn/codegen`：CMSIS-NN 代码生成模块，负责把 converter 输出转换为 `model.c`、`model.h`、权重文件、静态缓冲区规划和最小可编译工程。
- `src/nanoc_nn/pipeline`：一体化编排模块，负责串联 converter 与 codegen。
- `NanoC-NN-ONNX-Examples`：ONNX 测试模型样例集合，负责生成可供 converter 验证和教学使用的小模型。

整体流程为：先使用 ONNX 前端生成结构化中间表示，再由 CMSIS-NN codegen 生成 C 推理代码和工程骨架，最后在 PC 端和目标 Arm 工具链中完成编译、运行与 ONNX Runtime 精度对齐验证。

架构约束：`src/nanoc_nn/codegen` 的唯一上游是 `src/nanoc_nn/converter` 的标准输出目录。codegen 不直接读取原始 ONNX，也不在后端实现平行的 ONNX 解析逻辑；需要新增模型信息时，应升级 converter 的输出 schema。

## 项目目标

- 将至少一个指定 ONNX 模型转换为可编译、可运行的 CMSIS-NN C 推理工程。
- 面向 STM32、GD32 等 Cortex-M MCU 生成可嵌入用户固件工程的推理代码。
- 支持固定 shape、NCHW/NHWC 转换边界明确的 CNN 分类模型。
- 优先面向 CMSIS-NN 高价值路径：`Conv`、`DepthwiseConv`、`FullyConnected`、`Pooling`、`Activation`、`Softmax` 及常见逐元素算子。
- 生成代码不在运行期动态分配内存，激活缓冲区、权重、临时 buffer 均由生成阶段静态规划。
- C 端推理结果与 Python/ONNX Runtime 或量化参考实现完成精度对齐。

## 目录结构

```text
NanoC-NN/
├── AGENTS.md
├── pyproject.toml
├── src/
│   └── nanoc_nn/
│       ├── converter/
│       ├── codegen/
│       └── pipeline/
├── tests/
│   ├── converter/
│   ├── codegen/
│   └── pipeline/
├── docs/
│   ├── project/
│   ├── converter/
│   └── codegen/
├── tools/
├── third_party/
│   ├── README.md
│   └── CMSIS-NN/
├── NanoC-NN-ONNX-Examples/
└── README.md
```

## 一体化使用

推荐先在当前 conda 环境中安装根级工程的 editable 版本：

```bash
cd /Users/tiedan/Desktop/NanoC-NN

conda run -n nanoc-onnx-examples python -m pip install -e ".[dev]"
```

然后从统一 CLI 运行完整链路：

```bash
conda run -n nanoc-onnx-examples nanoc onnx-to-cmsis \
  --model /path/to/model.onnx \
  --target cortex-m4 \
  --sram-budget 128K \
  --flash-budget 512K
```

默认会在 ONNX 所在目录生成：

```text
model-nanoc-cmsis/
├── converter-output/
├── cmsis-codegen/
└── pipeline_report.md
```

也可以直接使用专用入口：

```bash
conda run -n nanoc-onnx-examples nanoc-onnx-to-cmsis \
  --model /path/to/model.onnx \
  --target cortex-m4
```

当前 `NanoC-NN-ONNX-Examples/` 会继续无损保留，后续样例体系再单独迭代。

## 开发约定

- 仓库协作规则见 `AGENTS.md`。
- 总体路线见 `docs/project/roadmap.md`，项目章程见 `docs/project/charter.md`。
- converter 计划见 `docs/converter/plan.md`，codegen 计划见 `docs/codegen/plan.md`。
- 文档优先使用中文。
- Python 工程目标版本为 Python 3.10+。
- 生成的 C 代码优先兼容 C99，并以 CMSIS-NN/CMSIS-Core 作为目标依赖。
- 生成工程需要清楚标注目标 MCU/内核、编译器、SRAM/Flash 预算和 CMSIS-NN 后端选项。
- 生成物、缓存、虚拟环境和构建产物不纳入版本管理。
