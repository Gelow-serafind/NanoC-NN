# CMSIS-NN ONNX 端侧代码生成器总体计划

## 1. 项目命名

建议项目中文名：

**CMSIS-NN ONNX 端侧代码生成器**

建议英文标识：

**CMSIS-NN ONNX Code Generator**

这个名称比“基于 CMSIS-NN 的 ONNX 代码生成器”更短，也更明确地表达了目标：输入 ONNX，输出面向 Arm Cortex-M/CMSIS-NN 的端侧推理 C 工程。

项目限定场景：

- 目标平台：Arm Cortex-M 系列低算力 MCU。
- 典型芯片：STM32、GD32 等使用 Cortex-M 内核的嵌入式平台。
- 运行环境：裸机或 RTOS。
- 典型约束：SRAM/Flash 有限、无动态内存、固定输入尺寸、批量大小通常为 1。
- 主要目标：帮助用户把小型神经网络部署到 MCU 固件中，而不是服务桌面端、Linux 或 Arm A-class 高性能推理。

## 2. 方向调整

早期路线是：

```text
ONNX 解析
    ↓
导出结构和权重
    ↓
人工手写 model.c
    ↓
调用自研 C 算子库
```

新的路线调整为：

```text
ONNX 模型
    ↓
ONNX Converter 生成中间表示
    ↓
CMSIS-NN Codegen 生成 C 推理工程
    ↓
调用 CMSIS-NN 优化内核
    ↓
Arm 工具链编译、MCU/FVP 运行、精度对齐
```

因此项目重点从“自研底层算子”切换为“代码生成、算子映射、量化参数、内存规划和验证闭环”。

## 3. 模块职责

### 3.1 nanoc_nn.converter

职责：

- 读取 ONNX 模型。
- 提取 graph、node、initializer、shape、权重和后续量化信息。
- 生成 `model_graph.json`、`model_summary.md`、`weights.h` 和转换报告。
- 为 codegen 提供稳定中间表示。

不负责：

- 直接生成 CMSIS-NN `model.c`。
- 做静态内存规划。
- 选择 CMSIS-NN API。

### 3.2 nanoc_nn.codegen

职责：

- 读取 converter 输出。
- 建立 ONNX 算子到 CMSIS-NN API 的映射。
- 生成 `model.c`、`model.h`、权重文件、静态 buffer 和工程骨架。
- 输出算子映射报告、内存报告和量化报告。
- 输出目标 MCU/内核、后端类型、SRAM/Flash 估算和移植说明。

上游约束：

- codegen 的唯一上游必须是 `nanoc_nn.converter`。
- codegen 只接受 converter 的标准输出目录，不接受原始 ONNX 作为直接输入。
- codegen 不重新解析 ONNX，不自行补齐图结构、shape、权重、layout 或量化信息。
- codegen 发现输入字段不足时，应提出 converter schema 需求，而不是在 codegen 内建立平行前端。

不负责：

- 自研完整神经网络算子库。
- 修改 CMSIS-NN 内核。
- 隐式处理不满足约束的模型。
- 生成依赖 Linux/Posix、文件系统、线程或堆内存的运行时代码。
- 直接生成 STM32CubeMX、Keil 或 GD32 官方 IDE 的完整项目文件。

### 3.3 NanoC-NN-ONNX-Examples

职责：

- 提供最小可复现 ONNX 样例。
- 为 converter 和 codegen 提供回归测试模型。
- 后续补充量化样例，服务 CMSIS-NN int8 路线。

## 4. 当前已完成

- 已建立 ONNX converter Python 包和命令行入口。
- 已支持模型加载、ONNX checker、shape inference、结构摘要、JSON 导出和 `weights.h` 导出。
- 已支持一批基础和真实模型高频算子结构解析：
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
- 已将参数 initializer 和辅助常量 initializer 分开处理。
- 已创建两个 ONNX 教学样例。
- 已将原自研 C operators 方向调整为 `nanoc_nn.codegen`，以 CMSIS-NN 作为底层推理内核。

## 5. 下一阶段里程碑

### M0：方向重置与文档同步

目标：

- 完成项目命名、章程、README、AGENTS 和计划文档更新。
- 明确项目不再自研完整 C 算子库，而是生成 CMSIS-NN C 工程。

验收标准：

- 文档中不再把自研 C operators 作为主路线。
- 目录结构统一使用根级 Python 工程，核心代码放在 `src/nanoc_nn/`。

### M1：中间表示契约固化

目标：

- 为 `model_graph.json` 增加 schema 版本。
- 明确 codegen 必需字段：节点、输入输出、shape、权重、常量、layout、opset、支持状态。
- 补测试防止 JSON 字段无意破坏。

验收标准：

- codegen 不需要重新读取 ONNX 文件即可获取第一版生成所需信息。
- codegen CLI 的输入语义明确为 converter 输出目录，而不是原始 ONNX 路径。

### M2：CMSIS-NN 算子映射设计

目标：

- 建立 `op_mapping.md`。
- 为 `Conv`、`Gemm/FullyConnected`、`Relu/Clip`、`MaxPool`、`AveragePool`、`Add`、`Softmax` 定义映射规则。
- 记录每个映射的 shape、layout、dtype、量化参数要求。
- 记录目标后端：scalar、DSP 或 MVE。

验收标准：

- 对一个模型能判断每层是否可映射到 CMSIS-NN。
- 不支持原因可读、可定位。

### M3：最小 Codegen 骨架

目标：

- 在 `src/nanoc_nn/codegen` 中建立 Python 包和命令行入口。
- 支持输入 converter 输出目录。
- 生成固定结构的 `generated/` 工程目录。
- 生成 MCU 友好的纯 C99 源文件，不依赖 HAL、OS 或堆内存。

验收标准：

- 能生成 `model.h`、`model.c`、`model_weights.h`、`main.c` 和 `CMakeLists.txt`。
- 生成文件能通过基础 C 语法检查。

### M4：最小 CNN 端到端生成

目标：

- 选择一个最小 CNN 样例。
- 将其 ONNX 解析结果输入 codegen。
- 生成第一版 CMSIS-NN 调用代码。

验收标准：

- 生成工程可编译。
- 每一层 C 调用都能追溯到 ONNX 节点。
- 输出报告明确列出 buffer、权重和算子映射。

### M5：静态内存规划

目标：

- 计算输入、输出、激活和临时 buffer 大小。
- 初期可采用保守双缓冲或逐层 buffer 策略。
- 生成 `memory_plan.md`。
- 生成 SRAM/Flash 估算，便于判断是否适合目标 STM32/GD32 型号。

验收标准：

- 运行期不使用 `malloc/free`。
- 所有 buffer 大小在生成阶段明确。
- 报告中明确标出激活、权重、scratch buffer 和代码/常量的估算来源。

### M6：量化路径接入

目标：

- 支持 ONNX Q/DQ 模型的 scale、zero point 提取。
- 生成 CMSIS-NN int8 调用所需量化参数。
- 补充一个量化 ONNX 样例。

验收标准：

- 缺少量化信息时明确失败，不隐式量化。
- int8 样例能走通解析、生成、编译和基础输出验证。

### M7：真实模型覆盖

目标：

- 重新建立临时真实 ONNX 模型池。
- 统计 converter 和 codegen 双侧 unsupported ops。
- 按频率和价值扩展支持。

验收标准：

- 输出覆盖报告。
- 明确下一批优先支持算子。

### M8：嵌入式目标验证

目标：

- 选择一个 Cortex-M 目标，例如 Cortex-M4、Cortex-M7、Cortex-M33 或 Cortex-M55。
- 使用 Arm GNU Toolchain、Arm Compiler 或 FVP 做编译/运行验证。
- 准备 STM32/GD32 类工程接入说明。

验收标准：

- 生成工程不依赖操作系统动态资源。
- 能说明如何接入用户固件主循环或任务。
- 报告中记录目标内核、编译器、优化等级和内存预算。

## 6. 下一步建议

当前最合适的下一步是 **M1：中间表示契约固化**。

原因：

- converter 已经能输出 `model_graph.json`，但它还不是正式 schema。
- codegen 如果直接开始写，很容易被 JSON 字段变动拖住。
- 先固化契约，后面生成 `model.c`、内存规划、量化参数和 MCU 内存预算都会更稳。
