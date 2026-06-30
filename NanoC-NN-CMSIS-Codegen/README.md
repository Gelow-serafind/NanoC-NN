# NanoC-NN-CMSIS-Codegen

CMSIS-NN 代码生成子项目。

该模块后续负责读取 `NanoC-NN-ONNX-Converter` 生成的 `model_graph.json`、权重资料和量化信息，生成面向 Arm CMSIS-NN 的 C 推理工程。

限定场景：本子项目主要面向 STM32、GD32 等 Arm Cortex-M 嵌入式低算力平台，用于在裸机或 RTOS 固件中部署小型神经网络。生成代码优先考虑固定输入、int8 量化、静态内存、有限 SRAM/Flash 和交叉编译，而不是桌面端、Linux 或高性能推理服务器。

上游约束：codegen 的唯一上游必须是 `NanoC-NN-ONNX-Converter`。本子项目只接受 converter 的标准输出目录，不直接接受原始 ONNX 文件；缺少字段时应推动 converter schema 升级，而不是在 codegen 内部重新解析 ONNX。

## 子项目定位

本目录不再自研完整神经网络算子库，而是以 CMSIS-NN 为底层内核依赖，重点实现以下能力：

- 将 ONNX 算子映射到 CMSIS-NN 可调用函数。
- 生成 `model.c`、`model.h`、权重文件和静态激活缓冲区规划。
- 生成最小 `main.c` 或验证入口，用于 PC/交叉编译 smoke test。
- 输出算子支持矩阵、量化参数报告和内存占用报告。
- 对不适合直接映射到 CMSIS-NN 的算子给出明确失败原因或 fallback 策略。
- 输出目标 MCU/内核、CMSIS-NN 后端、SRAM/Flash 预算和固件接入说明。

## 初期技术路线

1. 固定 converter 与 codegen 的中间表示字段，优先使用 `model_graph.json`。
2. 建立 ONNX 算子到 CMSIS-NN API 的映射表。
3. 优先生成静态 shape、单输入单输出 CNN 分类网络。
4. 优先处理 int8 量化模型；float32 模型先作为结构解析和参考验证输入，不作为 CMSIS-NN 加速主路径。
5. 生成代码不在运行期调用 `malloc` 或 `free`，所有工作区大小由生成阶段计算。
6. 生成代码不绑定 STM32 HAL、GD32 标准外设库或具体 IDE 工程，保持为可移植 C 源文件。

## CMSIS-NN 源码

当前已将 Arm 官方 CMSIS-NN 源码拉取到：

```text
third_party/CMSIS-NN/
```

说明：

- 来源：`ARM-software/CMSIS-NN` 的 `main` 分支源码包。
- License：Apache-2.0，详见 `third_party/CMSIS-NN/LICENSE`。
- 当前目录是 tarball 展开结果，不包含嵌套 `.git` 历史。
- CMSIS-NN 编译还需要 CMSIS-Core 路径，后续 codegen 命令会显式要求 `--cmsis-path`。
- CMSIS-NN 主路径是 TensorFlow Lite Micro 风格的 int8/int16 量化推理；float32/float16 API 属于实验路径。

## 目录结构

```text
NanoC-NN-CMSIS-Codegen/
├── Doc/
│   └── plan.md
├── nanoc_cmsis_codegen/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── generator.py
│   ├── loader.py
│   ├── mapper.py
│   ├── memory.py
│   ├── model.py
│   ├── quantization.py
│   ├── renderer.py
│   ├── report.py
│   └── schema.py
├── templates/
├── third_party/
│   ├── README.md
│   └── CMSIS-NN/
├── tests/
├── pyproject.toml
└── README.md
```

## 计划命令

后续计划提供如下入口：

```bash
python -m nanoc_cmsis_codegen \
  --input ../NanoC-NN-ONNX-Converter/build/export \
  --out build/generated \
  --cmsis-nn-root /path/to/CMSIS-NN \
  --cmsis-path /path/to/CMSIS \
  --backend mve \
  --target cortex-m4 \
  --sram-budget 128K \
  --flash-budget 512K
```

当前阶段只完成目录和计划设计，真正的 codegen 逻辑将在后续里程碑中逐步实现。

## 计划输出

```text
generated/
├── include/
│   ├── model.h
│   └── model_weights.h
├── src/
│   ├── model.c
│   └── main.c
├── reports/
│   ├── op_mapping.md
│   ├── memory_plan.md
│   ├── quantization.md
│   ├── target_report.md
│   └── source_notice.md
└── CMakeLists.txt
```

关键生成约束：

- 运行期主布局以 CMSIS-NN 的 NHWC 为准。
- 卷积优先生成 `arm_convolve_wrapper_s8`。
- 深度卷积优先生成 `arm_depthwise_conv_wrapper_s8`。
- 全连接优先生成 `arm_fully_connected_wrapper_s8`。
- 静态临时 buffer 使用 CMSIS-NN buffer size getter 或其 Python bindings 计算。
- BatchNormalization 初期只支持生成期折叠，不能折叠时阻塞生成。
- 生成报告必须给出 SRAM/Flash 估算，帮助判断能否部署到目标 STM32/GD32 型号。
- 生成的 `main.c` 仅用于 smoke test；真实固件中用户应调用 `model.h` 暴露的推理接口。
- 不生成完整 STM32CubeMX、Keil 或 GD32 IDE 工程，初期只提供纯 C 源码、CMake 示例和接入说明。

## 计划文档

技术路线和里程碑见 `Doc/plan.md`。
