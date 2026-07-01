# nanoc_nn.codegen 技术路线与里程碑

## 1. 模块目标

`nanoc_nn.codegen` 是 **CMSIS-NN ONNX 端侧代码生成器** 中的代码生成模块，目标是读取 `nanoc_nn.converter` 输出的中间表示和权重资料，生成基于 Arm CMSIS-NN 的 C 推理工程。

本模块的限定场景是 Arm 架构嵌入式低算力平台，典型目标包括 STM32、GD32 等 Cortex-M MCU。生成代码应服务 MCU 固件开发：资源有限、固定 shape、batch size 1、无运行期动态内存、可交叉编译、可审查、可插入裸机或 RTOS 工程。

本模块不重新解析 ONNX 文件，也不自研完整神经网络算子库。它的核心职责是：

- 消费 converter 生成的 `model_graph.json`、权重文件和后续量化资料。
- 将 ONNX 节点映射到 CMSIS-NN API 或生成期折叠逻辑。
- 生成 `model.c`、`model.h`、权重文件、静态 buffer、构建入口和验证入口。
- 输出算子映射、内存规划、量化参数和 unsupported ops 报告。
- 输出目标 MCU/内核、后端类型、SRAM/Flash 预算、编译器假设和移植注意事项。

## 2. 总体技术路线

整体采用 Python 实现，与 converter 保持相近的工程组织方式，使用命令行驱动生成。

推荐技术栈：

- Python 3.10+
- `argparse`：命令行入口。
- `json` / `dataclasses`：读取并表达 converter 中间表示。
- `pathlib`：路径处理。
- `pytest`：核心规则测试。
- C 端目标依赖：CMSIS-NN 与 CMSIS-Core。
- 可选：CMSIS-NN Python bindings，用于在生成阶段查询不同后端的 scratch buffer 大小。

核心流程如下：

```text
converter 输出目录
    ↓
读取 model_graph.json / weights.h / 量化资料
    ↓
校验 schema、shape、layout、dtype、opset 和支持状态
    ↓
ONNX 节点映射到 CMSIS-NN 调用或生成期折叠
    ↓
规划静态输入、输出、激活和临时 buffer
    ↓
生成 model.c / model.h / weights / main.c / CMakeLists.txt
    ↓
生成 op_mapping.md / memory_plan.md / quantization.md / codegen_report.txt
    ↓
C 语法检查、编译 smoke test、输出对齐验证
    ↓
Arm 工具链 / FVP / STM32-GD32 类目标验证
```

## 2.0 目标平台与运行环境

codegen 的生成物优先面向以下平台：

- Cortex-M0/M0+：仅能走 scalar C 路径，模型需要极小，性能预期保守。
- Cortex-M3：可作为基础兼容目标，但无 DSP/MVE 加速。
- Cortex-M4/M7/M33：优先利用 DSP extension，适合 STM32F4/F7/H7、GD32F4 等常见 MCU。
- Cortex-M55/M85：可利用 MVE/Helium，适合更高端边缘 AI MCU。

典型工程环境：

- 裸机 main loop。
- FreeRTOS / RTOS task。
- STM32CubeIDE / STM32CubeMX 生成工程。
- Keil MDK / Arm Compiler。
- Arm GNU Toolchain + CMake/Makefile。
- GD32 标准外设库或厂商 IDE 工程。

生成代码必须避免：

- `malloc` / `free`。
- 文件系统依赖。
- POSIX/Linux API。
- 线程、锁、动态加载。
- 隐式依赖 printf/stdout 作为推理功能的一部分。
- 绑定具体 HAL 或启动文件。

生成报告必须尽量帮助用户判断能否落到目标 MCU：

- 权重常量 Flash 估算。
- 激活 buffer SRAM 估算。
- CMSIS-NN scratch buffer SRAM 估算。
- 输入输出 buffer 大小。
- 目标 backend：scalar、DSP 或 MVE。
- 编译器和优化等级假设。

## 2.1 CMSIS-NN 源码走读结论

当前已将 Arm 官方 CMSIS-NN 源码拉取到：

```text
third_party/CMSIS-NN/
```

本地源码来自 `ARM-software/CMSIS-NN` 的 `main` 分支 tarball。当前目录是完整官方源码快照，不包含嵌套 `.git` 历史。本项目不得手工修改、裁剪或重排 CMSIS-NN 内容；需要升级时应整体替换官方快照，或改用 submodule / 外部依赖。

走读源码后，对 codegen 设计有以下直接影响：

- CMSIS-NN 主路径是 TensorFlow Lite Micro 风格的 `int8` / `int16` 量化推理，float32/float16 API 是实验性质，且默认 CMake 关闭。
- 公共头文件集中在 `Include/arm_nnfunctions.h` 和 `Include/arm_nn_types.h`。
- `cmsis_nn_dims` 使用 `n/h/w/c` 字段，主要 API 文档按 **NHWC** 表达激活张量。
- `arm_convolve_wrapper_s8`、`arm_depthwise_conv_wrapper_s8`、`arm_fully_connected_wrapper_s8` 会选择合适实现，codegen 初期应优先生成 wrapper 调用，而不是直接选最底层 kernel。
- 卷积和深度卷积使用 per-channel quant 参数：`cmsis_nn_per_channel_quant_params`，需要每个输出通道的 `multiplier` 和 `shift` 数组。
- Fully Connected wrapper 使用 `cmsis_nn_quant_params`，可表达 per-tensor 或 per-channel。
- Add、Mul、Softmax 需要 TFLM 风格的 multiplier、shift、offset、activation min/max、diff_min 等参数，不能只保存 ONNX scale/zero point。
- 每类需要临时内存的 API 都有 buffer size getter，例如 `arm_convolve_wrapper_s8_get_buffer_size`、`arm_depthwise_conv_wrapper_s8_get_buffer_size`、`arm_fully_connected_s8_get_buffer_size`、`arm_avgpool_s8_get_buffer_size`。
- CMSIS-NN 源码提供了可选 Python bindings，暴露 `convolve_wrapper_buffer_size` 等 host 端 buffer size 查询函数。后续可优先复用它，而不是在 Python 里手写所有 buffer 公式。
- CMSIS-NN CMake 需要 CMSIS-Core 路径，`Source/CMakeLists.txt` 中使用 `CMSIS_PATH` 并包含 `${CMSIS_PATH}/CMSIS/Core/Include`。
- CMSIS-NN 自身不把 host 编译作为主要支持目标，真正验证应优先面向 Arm GNU Toolchain / Arm Compiler / FVP；PC 端只能做有限语法和生成文件检查。
- BatchNormalization 在 int8 codegen 中不应作为运行期 BN 层优先生成，常规 CNN 推理应尽量在生成期折叠到 Conv/FC 权重、bias 和量化参数中。

## 3. 与 Converter 的边界

### 3.0 上游约束声明

`nanoc_nn.codegen` 的唯一上游必须是 `nanoc_nn.converter`。

codegen 只接受 converter 导出的标准产物目录作为输入，不接受原始 ONNX 文件作为直接输入，不在 codegen 内部重新解析 ONNX，也不绕过 converter 自行补充图结构、shape、权重或量化信息。所有模型结构、权重索引、layout、dtype、opset、量化参数和 unsupported ops 信息，都必须先由 converter 规范化后进入 `model_graph.json`、权重资料和转换报告。

如果 codegen 在映射 CMSIS-NN API、规划内存或生成 C 工程时发现字段不足，应把缺口反馈为 converter schema 需求，推动 converter 输出契约升级；不得在 codegen 中增加平行的 ONNX 前端逻辑。

### 3.1 输入契约

codegen 初期以 converter 输出目录作为输入，至少要求存在：

- `model_graph.json`：模型结构、节点、shape、initializer、权重角色和警告信息。
- `weights.h`：当前 converter 导出的 float32 参数权重头文件。
- `conversion_report.txt`：用于读取 converter 阶段的风险摘要。

后续为了 CMSIS-NN int8 主路径，需要 converter 继续补充：

- schema 版本。
- 量化参数：scale、zero point、multiplier、shift。
- layout 约束：ONNX 常见 NCHW 与 CMSIS-NN NHWC 的关系。
- constant 折叠结果：shape、axis、slice 参数等生成期常量。
- 权重导出策略：float32、int8、per-channel quantized weights 的不同输出。
- CMSIS-NN 权重布局：Conv 权重需要从 ONNX 常见 `O,I,H,W` 转换到 CMSIS-NN 文档使用的 `C_OUT,H,W,C_IN`；depthwise 权重需要转换到 `1,H,W,C_OUT` 或对应 wrapper 要求。
- bias 数据类型：int8 卷积/FC 常用 int32 bias，需要能从量化参数校验或生成。

### 3.2 责任划分

- converter 负责：ONNX 解析、shape 推断、属性归一化、权重提取、量化信息提取。
- codegen 负责：CMSIS-NN 映射、C 文件生成、内存规划、构建文件生成、报告生成。
- examples 负责：提供可复现模型、输入数据和参考输出。

codegen 不应重新读取 ONNX 文件。若发现中间表示缺字段，应向 converter 提出 schema 需求，而不是在 codegen 中绕过前端。

### 3.3 布局策略

codegen 内部以 CMSIS-NN 的 NHWC 作为运行期主布局。

初期策略：

- converter 可继续报告 ONNX 原始 shape，但必须标注原始 layout。
- codegen 生成 `cmsis_nn_dims` 时使用 NHWC。
- 对 NCHW ONNX 模型，优先在生成期重排权重；激活 tensor 是否插入 transpose 需要显式记录。
- 若某条路径不能安全确认 NCHW/NHWC 关系，strict 模式下直接失败。

## 4. 当前目录结构

```text
NanoC-NN/
├── pyproject.toml
├── src/
│   └── nanoc_nn/
│       └── codegen/
│           ├── __init__.py
│           ├── __main__.py
│           ├── cli.py
│           ├── loader.py
│           ├── model.py
│           ├── schema.py
│           ├── mapper.py
│           ├── memory.py
│           ├── generator.py
│           ├── quantization.py
│           ├── renderer.py
│           ├── report.py
│           └── templates/
├── tests/
│   └── codegen/
├── docs/
│   └── codegen/
└── third_party/
    └── CMSIS-NN/
```

目录职责：

- `src/nanoc_nn/codegen/`：代码生成器 Python 包。
- `src/nanoc_nn/codegen/templates/`：C 工程模板。初期使用简单占位模板，不引入模板框架。
- `tests/codegen/`：schema 校验、算子映射、命名、内存规划和生成结果测试。
- `docs/codegen/`：技术路线、里程碑和 CMSIS-NN 映射设计文档。
- `third_party/CMSIS-NN/`：Arm 官方 CMSIS-NN 完整源码快照。

## 5. 主要模块设计

### 5.1 `cli.py`

提供命令行入口：

```bash
python -m nanoc_nn.codegen \
  --input build/export \
  --out build/generated \
  --cmsis-nn-root /path/to/CMSIS-NN \
  --cmsis-path /path/to/CMSIS \
  --backend mve \
  --target cortex-m4
```

计划参数：

- `--input`：converter 输出目录。
- `--out`：生成工程目录。
- `--cmsis-nn-root`：CMSIS-NN 源码或安装路径。
- `--cmsis-path`：CMSIS-Core 所在仓库或安装路径，用于提供 `CMSIS/Core/Include`。
- `--cmsis-version`：目标 CMSIS-NN 版本标注。
- `--target`：目标 Cortex-M 类型，例如 `cortex-m0`、`cortex-m3`、`cortex-m4`、`cortex-m7`、`cortex-m33`、`cortex-m55`。
- `--backend`：buffer size 和映射策略使用的后端，可选 `scalar`、`dsp`、`mve`；可根据 `--target` 自动推断。
- `--sram-budget`：目标 MCU 可用于模型的 SRAM 预算，例如 `64K`、`256K`。
- `--flash-budget`：目标 MCU 可用于模型权重和代码的 Flash 预算。
- `--project-style`：生成工程风格，初期可选 `cmake`，后续扩展 `keil`、`stm32cubeide`、`makefile`。
- `--strict`：遇到 unsupported ops、动态 shape、缺量化参数时直接失败。
- `--verbose`：输出详细映射和生成日志。

### 5.2 `loader.py` 与 `schema.py`

负责读取和校验 converter 产物：

- 检查 `model_graph.json` 是否存在。
- 检查 schema 版本是否在支持范围内。
- 检查节点、initializer、shape、layout、warnings 字段完整性。
- 将 JSON 转换为 codegen 内部 dataclass。
- 校验 converter 输出是否包含 codegen 必需的 layout 和量化字段；缺失时标记为 `blocked`，而不是伪造默认值。

### 5.3 `mapper.py`

负责 ONNX 到 CMSIS-NN 的映射判断。

初期映射优先级：

- `Conv` -> `arm_convolve_wrapper_s8`，当 `group == in_channels == out_channels / ch_mult` 时映射到 `arm_depthwise_conv_wrapper_s8`。
- `Gemm` / `MatMul` -> `arm_fully_connected_wrapper_s8`。
- `Relu` / `Clip` -> 优先折叠到 Conv/FC/Pool 的 `cmsis_nn_activation`，无法融合时再考虑独立 activation。
- `MaxPool` -> `arm_max_pool_s8`。
- `AveragePool` / `GlobalAveragePool` -> `arm_avgpool_s8`，GlobalAveragePool 映射为 kernel 覆盖完整 H/W。
- `Add` -> `arm_elementwise_add_s8`。
- `Mul` -> `arm_elementwise_mul_s8`。
- `Softmax` -> `arm_softmax_s8`。
- `Concat` -> `arm_concatenation_s8_x/y/z/w`，轴映射必须基于 NHWC。
- `Reshape` -> 无数据重排时生成期折叠；需要拷贝时使用 `arm_reshape_s8`。
- `Transpose` -> 若是布局桥接且可消除则生成期折叠，否则使用 `arm_transpose_s8`。
- `Constant` / `Cast` / `Shape` / `Gather` / `Slice` / `Unsqueeze` -> 优先作为生成期 shape/参数计算，不进入运行期。
- `BatchNormalization` -> int8 推理中优先生成期折叠到 Conv/FC；不能折叠则标记 unsupported。

每个映射结果需要包含：

- ONNX 节点 ID 和名称。
- 目标 CMSIS-NN API、融合动作或生成期折叠动作。
- 输入输出 shape。
- dtype 与量化要求。
- 是否需要 layout 转换。
- 是否需要临时 buffer。
- 不支持原因。

映射状态建议：

- `direct_api`：可直接生成 CMSIS-NN API 调用。
- `wrapper_api`：使用 CMSIS-NN wrapper API 选择最优实现。
- `fused`：被融合到前后层参数中，例如 Relu/Clip。
- `folded`：生成期常量或 shape 计算，不生成运行时代码。
- `layout_transform`：需要显式布局转换。
- `blocked`：理论可支持，但当前缺少 schema、量化或 layout 信息。
- `unsupported`：当前不计划支持或 CMSIS-NN 无合适路径。

### 5.4 `quantization.py`

负责把 converter 提供的量化信息转换成 CMSIS-NN 调用参数。

需要生成或校验：

- input/output/filter zero point 到 CMSIS-NN offset 的转换：通常为 `-zero_point`。
- Conv / depthwise Conv 的 per-channel `multiplier[]`、`shift[]`。
- FC 的 per-tensor 或 per-channel quant 参数。
- Add/Mul 的 input multiplier、input shift、left shift、output multiplier、output shift。
- Softmax 的 `mult`、`shift`、`diff_min`。
- activation min/max。
- int32 bias 与量化尺度关系。

缺少这些字段时，codegen 只能生成结构报告，不能生成声称可运行的 CMSIS-NN 调用代码。

### 5.5 `memory.py`

负责静态内存规划：

- 输入 buffer。
- 输出 buffer。
- 激活 buffer。
- 每层临时 buffer。
- CMSIS-NN context buffer。
- Flash 中的权重、bias、量化表和常量表估算。

初期策略可以保守：

- 单输入单输出模型。
- 固定 batch size 1。
- 双激活缓冲区轮换。
- 每层临时 buffer 取最大需求值。
- 使用 CMSIS-NN buffer size getter 计算 `cmsis_nn_context` scratch 大小。
- 后端选择影响 buffer size，`scalar`、`dsp`、`mve` 需要分开记录。
- 若用户提供 `--sram-budget` 或 `--flash-budget`，超过预算时 strict 模式失败，非 strict 模式报告高风险。

后续再做生命周期分析和 buffer 复用优化。

### 5.6 `generator.py` 与 `renderer.py`

负责生成文件：

- `include/model.h`
- `include/model_weights.h`
- `src/model.c`
- `src/main.c`
- `CMakeLists.txt`

初期模板只生成结构清晰、可检查的 C 代码；在 CMSIS-NN API 参数全部确定前，可以先生成带 `TODO` 和明确失败信息的骨架代码，但不得声称该层已可运行。

生成工程需要包含：

- CMSIS-NN include 路径。
- CMSIS-Core include 路径。
- 目标 `TARGET_CPU` 或等价编译参数说明。
- 可配置的 CMSIS-NN CMake 选项，例如是否开启 float API。
- 便于嵌入 STM32/GD32 工程的纯 C 源文件，不依赖 HAL。
- 可选的最小 `main.c` 只用于 smoke test，真实固件中用户可直接调用 `model.h` 暴露的接口。

### 5.7 `report.py`

负责生成报告：

- `codegen_report.txt`：总体成功/失败、输入输出和警告。
- `op_mapping.md`：每个 ONNX 节点到 CMSIS-NN 的映射。
- `memory_plan.md`：静态 buffer 和临时 buffer 规划。
- `quantization.md`：量化参数、缺失项和风险。
- `unsupported_ops.md`：不支持算子和原因。
- `source_notice.md`：记录使用的 CMSIS-NN 来源、版本、license 和是否 vendored/submodule。
- `target_report.md`：目标内核、后端、编译器、SRAM/Flash 预算和移植注意事项。

## 6. 初期支持边界

初期只面向：

- 固定 shape。
- batch size 1。
- 单输入、单输出 CNN 分类模型。
- 优先 int8 量化模型，遵循 TFLM/CMSIS-NN 风格。
- 支持边界清晰的 NCHW/NHWC layout，并在运行期以 NHWC 为主。
- 无控制流、无动态 shape、无动态 sequence。
- STM32、GD32 等 Cortex-M MCU 固件场景。
- 裸机或 RTOS 环境。

初期不实现：

- 全量 ONNX 算子兼容。
- 自动训练或量化。
- 运行期动态内存分配。
- CMSIS-NN 内核改写。
- 多后端代码生成。
- 对 unsupported ops 的静默 fallback。
- 隐式 float32 到 int8 量化。
- 忽略 layout 的权重或激活直接复用。
- Linux/Android/桌面端推理。
- 自动生成完整 STM32CubeMX、Keil 或 GD32 IDE 工程。
- 自动选择具体芯片型号、时钟、外设和链接脚本。

## 7. 里程碑清单

### M0：目录骨架与计划文档

目标：

- 按根级 Python 工程风格建立 `src/nanoc_nn/codegen` 目录结构。
- 编写 `docs/codegen/plan.md`。
- 更新 README，明确使用方式和当前边界。
- 拉取 CMSIS-NN 源码并记录来源、license 和本地路径。
- 基于 CMSIS-NN 源码修订 codegen plan。

验收标准：

- 目录结构清晰。
- plan 能指导后续逐步编码。
- 不再出现 codegen 是 C operators 的旧定位。
- `third_party/CMSIS-NN/` 存在源码，并明确是否纳入版本管理。

### M1：命令行入口与输入目录检查

目标：

- 在根级 `pyproject.toml` 中注册 Python 包和 CLI 入口。
- 实现 `python -m nanoc_nn.codegen --input --out` 的入口。
- 检查 converter 输出目录是否存在必要文件。
- 增加目标平台参数：`--target`、`--backend`、`--sram-budget`、`--flash-budget`。

验收标准：

- 输入目录不存在时给出明确错误。
- 缺少 `model_graph.json` 时给出明确错误。
- 能创建输出目录和基础报告。
- `codegen_report.txt` 中记录目标 MCU/内核和预算参数。

### M2：Schema 校验与内部模型

目标：

- 读取 `model_graph.json`。
- 定义 codegen 内部 dataclass。
- 校验 inputs、outputs、nodes、initializers、warnings 等字段。
- 输出 schema 校验报告。
- 增加 layout、dtype、quantization、weight_layout、bias_type 等 codegen 必需字段检查。

验收标准：

- 能加载当前 converter 生成的 JSON。
- 缺字段或字段类型错误时能定位到具体字段。
- 单元测试覆盖正常和异常 JSON。
- 缺少量化字段时状态为 `blocked`，不是 `supported`。

### M3：ONNX 到 CMSIS-NN 映射表

目标：

- 建立第一版映射表和支持状态。
- 输出 `op_mapping.md`。
- 对 unsupported ops 给出原因。
- 初期优先覆盖 CMSIS-NN s8 API：Conv、DepthwiseConv、FullyConnected、Add、Mul、MaxPool、AvgPool、Softmax、Concat、Reshape、Transpose。

验收标准：

- 对每个节点产生 `wrapper_api`、`direct_api`、`fused`、`folded`、`layout_transform`、`unsupported` 或 `blocked` 状态。
- 能统计模型是否可生成。
- 支持状态不会与 converter 的 supported 状态混淆。

### M4：最小工程生成骨架

目标：

- 生成固定目录结构：

```text
generated/
├── include/
├── src/
├── reports/
└── CMakeLists.txt
```

- 生成可读的 `model.h`、`model.c`、`main.c`、`model_weights.h`。
- 生成 `target_report.md`，说明如何接入 STM32/GD32 类固件工程。

验收标准：

- 文件完整。
- C 文件可以被 C99 编译器做语法检查。
- 报告中明确标注哪些代码只是骨架，哪些层已映射可运行。
- 生成代码不包含 HAL、OS、文件系统或堆内存依赖。

### M5：静态内存规划

目标：

- 根据 shape 计算输入、输出和激活 buffer 大小。
- 生成 `memory_plan.md`。
- 在 `model.c` 中生成静态 buffer 声明。
- 调用 CMSIS-NN buffer size getter 或 Python binding 计算每层 scratch buffer。
- 根据 `--sram-budget` 和 `--flash-budget` 生成预算判断。

验收标准：

- 不使用 `malloc/free`。
- buffer 大小可追溯到具体张量。
- 动态 shape 模型在 strict 模式下失败。
- `memory_plan.md` 标明目标 backend：scalar、dsp 或 mve。
- 超预算情况必须在报告中高亮。

### M6：最小 CNN 可运行代码生成

目标：

- 选择一个 converter 已支持的小型 CNN。
- 生成 CMSIS-NN 调用代码。
- 生成最小 smoke test。
- 若样例是 float32，先标记为结构可生成但 CMSIS-NN int8 路径 blocked；真正可运行闭环优先选择 int8/QDQ 样例。

验收标准：

- 工程能编译。
- 每一层调用都能追溯到 ONNX 节点。
- 若缺少量化参数，报告明确说明阻塞点。

### M7：量化参数接入

目标：

- 消费 converter 输出的 Q/DQ 量化信息。
- 生成 CMSIS-NN int8 调用所需参数。
- 输出 `quantization.md`。
- 实现 per-channel multiplier/shift、offset、activation min/max、Softmax diff_min 等 CMSIS-NN 参数生成或读取。

验收标准：

- 缺少 scale、zero point、multiplier、shift 时不能静默继续。
- 最小 int8 样例能生成完整参数。
- 与 TFLM/CMSIS-NN 量化语义对齐。

### M8：端到端验证

目标：

- 生成可运行 C 工程。
- 对比 ONNX Runtime 或 Python 参考输出。
- 生成验证报告。
- 使用 Arm 工具链或 FVP 验证 Cortex-M 目标构建。

验收标准：

- 至少一个小模型完成端到端生成、编译、运行和输出对齐。
- 报告中记录输入、输出、误差和平台信息。
- 平台信息包含目标内核、编译器、优化等级、后端类型和内存预算。

### M9：STM32/GD32 类工程接入验证

目标：

- 输出适合复制进 STM32CubeIDE、Keil MDK、CMake/Makefile 或 GD32 固件工程的源文件。
- 编写接入说明：初始化、输入 buffer 填充、调用 `nanoc_model_run`、读取输出。
- 验证生成代码不依赖厂商 HAL 或标准外设库。

验收标准：

- 至少给出一个 STM32/GD32 类固件工程接入文档。
- 生成接口边界清晰，用户可以在主循环或 RTOS task 中调用。
- 不要求 codegen 直接生成完整 IDE 工程。

### M10：真实模型覆盖扩展

目标：

- 使用开源 ONNX 模型池轮询 converter + codegen。
- 统计 unsupported ops、缺量化参数、layout 问题和内存问题。
- 按出现频率和实现价值扩展支持。

验收标准：

- 输出批量覆盖报告。
- 明确下一批优先支持项。

## 8. 风险与处理策略

| 风险 | 影响 | 策略 |
| :--- | :--- | :--- |
| converter JSON schema 不稳定 | codegen 易碎 | 先做 schema 版本和契约测试。 |
| ONNX NCHW 与 CMSIS-NN NHWC 不一致 | 生成代码错误 | codegen 内部以 NHWC 为主，layout 必须显式记录，必要时生成 transpose 或拒绝。 |
| float32 模型缺少量化参数 | 无法使用 CMSIS-NN 高价值路径 | 初期报告 blocked，后续支持 Q/DQ 模型。 |
| CMSIS-NN API 版本变化 | 生成代码不兼容 | 显式记录目标 CMSIS-NN 版本，后续做适配层。 |
| 静态内存规划错误 | C 端越界 | 使用 CMSIS-NN buffer size getter，生成 memory plan 和尺寸断言，先用保守策略。 |
| CMSIS-Core 路径缺失 | 工程无法编译 | CLI 显式要求 `--cmsis-path` 或生成报告说明缺失。 |
| BatchNormalization 运行期映射不清晰 | int8 模型错误或无法生成 | 初期只支持生成期折叠；不能折叠则 blocked。 |
| 目标 MCU SRAM/Flash 不足 | 工程可生成但无法部署 | 支持 `--sram-budget` / `--flash-budget`，生成预算报告和超预算阻塞。 |
| 厂商工程差异大 | 自动生成 IDE 工程成本高 | 初期只生成纯 C 源码、CMake 示例和接入说明，不绑定 HAL/IDE。 |
| 低端 Cortex-M0/M3 性能不足 | 推理延迟不可接受 | target report 中记录后端能力和性能风险，优先支持 M4/M7/M33/M55。 |

## 9. 下一步执行建议

完成本计划后，建议从 M1 开始编码：

1. 建立 `pyproject.toml` 和命令行入口。
2. 实现输入目录检查。
3. 读取 `model_graph.json`。
4. 输出最小 `codegen_report.txt`。

这一步不需要立刻生成 CMSIS-NN 调用代码，先把 codegen 的输入边界守住。

## 10. 第一版执行记录

当前已经从 M0 推进到 M10 的第一版闭环，完成内容如下：

- M0-M1：目录、README、CLI、目标平台参数和 CMSIS-NN 源码说明已经建立。
- M2：可读取 converter 的 `model_graph.json`，并输出 schema warning/error。
- M3：已生成第一版 ONNX 到 CMSIS-NN API 的映射报告。
- M4：已生成 `include/`、`src/`、`reports/` 和 `CMakeLists.txt`。
- M5：已输出输入、输出、双激活 buffer、scratch 和 Flash/SRAM 预算估算。
- M6：已生成可 C99 smoke compile 的 C 工程骨架；缺量化时不伪装为可运行 int8 推理。
- M7：已生成量化缺口报告；当前 converter 无量化 section 时 runtime 层状态为 `blocked`。
- M8：本地 smoke compile 已接入验证脚本。
- M9：已生成 STM32/GD32 类固件接入说明 `reports/firmware_integration.md`。
- M10：已用多种本地生成 ONNX 执行 converter -> codegen -> C99 编译覆盖测试。

当前验证命令：

```bash
conda run -n nanoc-onnx-examples python tools/validate_local_models.py
```

当前批量覆盖报告：

```text
build/local-validation/coverage_report.md
```

第一版结论：

- codegen 已能稳定消费 converter 输出目录，未绕过 converter 直接解析 ONNX。
- 生成工程骨架和报告闭环已经跑通。
- 由于 converter 尚未输出 int8 量化信息，CMSIS-NN runtime 调用仍处于 `blocked`。
- 下一阶段应优先补 converter schema version、量化参数、layout 规范和 Q/DQ 模型解析。
