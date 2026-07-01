# 项目章程

## 1. 项目基本信息

- **项目名称：** CMSIS-NN ONNX 端侧代码生成器
- **英文标识：** CMSIS-NN ONNX Code Generator
- **项目定位：** 面向 Arm Cortex-M 低算力 MCU 的 ONNX 到 CMSIS-NN C 推理工程生成工具
- **仓库代号：** NanoC-NN

## 2. 项目背景与目的

### 2.1 背景

项目早期目标是自研一套轻量级 C 神经网络算子库，并由 ONNX 解析工具输出结构和权重，再人工编写 `model.c` 拼接推理流程。这个路线有助于理解底层算子，但在真实嵌入式部署中存在三个明显问题：

- 自研卷积、全连接、池化等算子很难在短期内达到成熟库的性能和稳定性。
- 复杂模型依靠人工手写 `model.c` 成本高、易错，难以规模化。
- Arm Cortex-M 生态中已经存在 CMSIS-NN 这类成熟优化内核，继续重复实现基础算子性价比不高。

CMSIS-NN 是 Arm 面向 Cortex-M 的神经网络内核库，具备针对 MCU 的优化实现。因此项目方向调整为：不再把“自研 C 算子库”作为主目标，而是以 CMSIS-NN 为执行后端，重点开发 ONNX 到 CMSIS-NN C 工程的代码生成能力。

本项目限定的主要使用场景是 STM32、GD32 等 Arm Cortex-M 系列 MCU 上的嵌入式神经网络开发。这类平台通常算力有限、SRAM/Flash 预算紧张、运行环境可能是裸机或 RTOS，并且工程需要接入 Keil/Arm Compiler、Arm GNU Toolchain、CMake 或厂商 IDE。项目设计必须优先考虑静态内存、固定 shape、可交叉编译、可审查生成代码和 int8 量化推理。

### 2.2 目的

开发一套可解释、可验证、可逐步扩展的 ONNX 代码生成工具链：

```text
ONNX 模型
    ↓
ONNX 前端解析与规范化
    ↓
中间表示 model_graph.json / 权重 / 量化信息
    ↓
CMSIS-NN Codegen
    ↓
model.c / model.h / weights / memory plan / build files
    ↓
PC 与 Cortex-M 工具链编译验证
```

项目最终目标是让用户输入一个受支持边界内的 ONNX 模型，即可生成基于 CMSIS-NN 的 C 推理工程，并获得算子映射、内存规划、量化参数和验证报告。

生成工程的第一目标不是在 PC 上高性能运行，而是能被移植到 STM32/GD32 等 Cortex-M 固件工程中，作为用户应用代码的一部分完成端侧推理。

## 3. 项目范围

### 3.1 包含在范围内

#### ONNX 前端解析工具

- 解析指定 ONNX 模型文件的计算图结构。
- 提取输入输出、节点顺序、算子属性、shape、initializer 和权重数据。
- 识别并导出与 CMSIS-NN codegen 相关的量化信息。
- 输出 `model_graph.json`、结构摘要、转换报告和权重资料。

#### CMSIS-NN 代码生成器

- 读取 ONNX 前端输出的中间表示。
- 将受支持 ONNX 算子映射到 CMSIS-NN API。
- 生成 `model.c`、`model.h`、权重文件、静态激活缓冲区和临时 buffer 规划。
- 生成最小可编译工程骨架，例如 `CMakeLists.txt`、`main.c` 或 smoke test 入口。
- 输出算子映射报告、内存占用报告和量化参数报告。
- 输出目标 MCU、目标内核、编译器、CMSIS-NN 后端、SRAM/Flash 估算和移植注意事项。

codegen 的唯一上游必须是 `nanoc_nn.converter`。codegen 只消费 converter 输出的标准产物目录，不直接读取原始 ONNX，不在内部重新实现 ONNX 前端解析；如果生成阶段需要额外字段，应通过 converter schema 升级解决。

#### 样例与验证

- 维护小型 ONNX 样例，用于验证解析、代码生成和端到端推理流程。
- 在 PC 端做语法、链接和基础运行验证。
- 使用 Arm GCC/Arm Compiler/FVP 或目标开发板做阶段性嵌入式侧验证。
- 与 Python/ONNX Runtime 或量化参考实现做输出对齐。
- 对真实开源 ONNX 模型做阶段性覆盖测试，统计 unsupported ops 和转换风险。

### 3.2 不在范围内

- 自研完整 C 神经网络算子库。
- 全量 ONNX 算子兼容。
- 无限制动态图、动态 shape 或多输入复杂控制流模型。
- 自动训练、量化感知训练或模型压缩流程。
- 针对非 Arm Cortex-M 后端的专用代码生成。
- 面向 Linux/Android/桌面端/Arm A-class 处理器的高性能推理框架。
- 自动适配具体厂商 HAL、外设、启动文件、链接脚本或 IDE 工程格式。
- 对 CMSIS-NN 内核本身做汇编级或算法级改写。

## 4. 当前阶段成果

当前已经完成或初步完成：

- 仓库已调整为根级 Python 工程，核心能力拆分为 `nanoc_nn.converter`、`nanoc_nn.codegen` 和 `nanoc_nn.pipeline` 三个模块。
- `nanoc_nn.converter` 已具备命令行入口、ONNX 加载校验、shape 推断、结构摘要、JSON 导出和 `weights.h` 导出能力。
- converter 已支持 `Conv`、`Relu`、`Gemm`、`MatMul`、`MaxPool`、`Softmax`、`Flatten`、`Reshape`、`Add`、`Constant`、`Transpose`、`Cast`、`BatchNormalization`、`GlobalAveragePool` 的结构解析。
- converter 已能区分参数 initializer 与辅助常量 initializer，避免把 `Reshape` shape 常量错误导出为 C 权重。
- `NanoC-NN-ONNX-Examples` 已包含两个教学样例，用于生成和验证小型 ONNX。

## 5. 高层级需求

| 编号 | 需求描述 | 优先级 |
| :--- | :--- | :--- |
| REQ-01 | 工具链支持通过命令行指定 ONNX 模型路径和输出目录。 | 高 |
| REQ-02 | converter 输出稳定、可版本化的 `model_graph.json`，作为 codegen 输入契约。 | 高 |
| REQ-03 | codegen 能生成基于 CMSIS-NN 的 `model.c`、`model.h`、权重和最小工程骨架。 | 高 |
| REQ-04 | 生成代码不在运行期使用动态内存分配。 | 高 |
| REQ-05 | 工具链必须输出算子映射、buffer 大小、量化参数和 unsupported ops 报告。 | 高 |
| REQ-06 | 初期优先支持固定 shape、CNN 分类模型和 int8 量化路径。 | 高 |
| REQ-07 | 对 CMSIS-NN API 版本、张量布局和量化约束进行显式记录。 | 中 |
| REQ-08 | codegen 必须面向 STM32/GD32 等 Cortex-M MCU 的静态内存部署方式，不生成依赖操作系统动态资源的运行时代码。 | 高 |
| REQ-09 | 生成报告必须给出 SRAM/Flash 占用估算、目标内核和编译器假设。 | 高 |
| REQ-10 | codegen 的唯一上游必须是 converter 标准输出目录，不得直接解析原始 ONNX 或建立平行前端。 | 高 |

## 6. 主要可交付成果

1. **ONNX 前端解析工具**：Python 包、命令行入口、测试和文档。
2. **中间表示规范**：`model_graph.json` 字段说明、版本号和兼容策略。
3. **CMSIS-NN Codegen**：生成 `model.c`、`model.h`、权重文件、静态内存规划和构建入口。
4. **样例模型工程**：至少一个从 ONNX 到 CMSIS-NN C 工程的端到端示例。
5. **目标 MCU 移植说明**：记录如何接入 STM32/GD32 这类 Cortex-M 固件工程。
6. **验证报告**：记录 ONNX Runtime/参考实现与 C 端输出对比、算子覆盖和残留风险。

## 7. 总体里程碑计划

| 里程碑 | 描述 | 当前状态 |
| :--- | :--- | :--- |
| M0：项目方向调整 | 明确项目从自研 C 算子库转向 CMSIS-NN ONNX 代码生成。 | 进行中 |
| M1：ONNX 前端基础闭环 | 完成模型解析、shape 推断、结构摘要、权重导出和基础测试。 | 已完成 |
| M2：真实模型解析扩展 | 支持真实 CNN 中高频结构算子，形成 unsupported ops 统计方法。 | 已部分完成 |
| M3：中间表示契约固化 | 为 codegen 固定 `model_graph.json` schema、权重索引和量化字段。 | 待开始 |
| M4：CMSIS-NN 映射设计 | 建立 ONNX 算子到 CMSIS-NN API 的映射表和约束检查。 | 待开始 |
| M5：最小代码生成闭环 | 针对一个小型模型生成 `model.c`、`model.h`、权重和可编译工程。 | 待开始 |
| M6：静态内存规划 | 生成激活缓冲区、临时 buffer、输入输出 buffer 的静态分配方案。 | 待开始 |
| M7：量化路径接入 | 支持 int8 量化参数提取、权重整理和 CMSIS-NN 量化调用参数生成。 | 待开始 |
| M8：端到端验证 | C 工程与 ONNX Runtime/参考实现完成输出对齐。 | 待开始 |
| M9：嵌入式平台移植验证 | 在 Arm 工具链、FVP 或 STM32/GD32 类开发板上验证生成工程。 | 待开始 |
| M10：模型覆盖扩展 | 用真实开源模型持续统计缺口并扩展支持边界。 | 待开始 |

## 8. 技术路线重点

### 8.1 Converter 到 Codegen 的边界

converter 不直接拼接 CMSIS-NN 调用代码，而是生成稳定的中间表示。codegen 的唯一上游是 converter 的结构化输出目录；codegen 不重新解析 ONNX，不接受原始 ONNX 作为直接输入，也不在后端补写图解析逻辑。若 codegen 映射 CMSIS-NN API 时发现字段不足，应把缺口沉淀为 converter schema 需求。

### 8.2 CMSIS-NN 优先级

优先映射以下模式：

- `Conv` / depthwise convolution
- `FullyConnected` / `Gemm`
- `MaxPool` / `AveragePool`
- `Relu` / `Clip`
- `Add` / `Mul`
- `Softmax`
- 常见 shape 辅助算子在生成期折叠，不进入运行期

### 8.3 量化策略

CMSIS-NN 的高价值路径主要来自量化推理。项目后续需要从 float32 结构导出升级到量化信息提取与校验：

- 优先支持 ONNX Q/DQ 形式的量化模型。
- 记录 scale、zero point、multiplier、shift 等生成 CMSIS-NN 调用需要的参数。
- 对缺失量化信息的 float32 模型，先报告为非 CMSIS-NN 加速主路径，不做隐式量化。

### 8.4 代码生成策略

生成代码应遵守：

- 不动态分配内存。
- 不隐藏 layout 变换。
- 每一层生成清晰的调用块和错误码。
- 生成报告能反查每个 C 调用来自哪个 ONNX 节点。
- 对 unsupported ops 直接失败或明确标注 fallback，不静默跳过。
- 输出代码应便于复制进 STM32CubeIDE、Keil MDK、Eclipse Embedded CDT、CMake/Makefile 等 MCU 工程。
- 不依赖文件系统、线程、标准输入输出、堆内存或 Linux/Posix API。

## 9. 主要风险与应对

| 风险项 | 影响 | 概率 | 应对策略 |
| :--- | :--- | :--- | :--- |
| ONNX 与 CMSIS-NN 算子语义不完全一致 | 高 | 中 | 建立逐算子映射表，记录 layout、padding、stride、量化约束和失败条件。 |
| float32 模型缺少 CMSIS-NN 所需量化参数 | 高 | 高 | 初期明确 int8 量化路径优先，float32 仅作为结构解析和参考验证输入。 |
| 静态内存规划错误导致覆盖或越界 | 高 | 中 | 生成 memory plan，增加 buffer 尺寸断言和 PC 端 sanitizer/语法检查。 |
| CMSIS-NN API 版本变化 | 中 | 中 | 在 codegen 配置中显式记录目标 CMSIS-NN 版本，建立适配层。 |
| 真实模型 unsupported ops 增长 | 中 | 高 | 用模型库轮询测试统计频率，按价值和实现成本分批支持。 |
| MCU SRAM/Flash 超预算 | 高 | 高 | 生成 memory plan 和 size report，超过预算时阻塞或警告。 |
| 厂商工程差异导致移植困难 | 中 | 中 | 生成尽量纯净的 C99 源文件和 CMake 示例，不绑定 STM32 HAL/GD32 标准库。 |

## 10. 当前下一步建议

下一步优先进入 `nanoc_nn.codegen`：

1. 定义 codegen 输入契约：读取 converter 的 `model_graph.json` 和权重资料。
2. 建立第一版 ONNX 到 CMSIS-NN 的算子映射表。
3. 选择一个最小 CNN 样例作为端到端目标。
4. 生成最小 `model.c`、`model.h`、权重文件和 CMake 工程。
5. 先完成编译闭环，再推进量化参数、内存预算和 STM32/GD32 类 MCU 移植验证。
