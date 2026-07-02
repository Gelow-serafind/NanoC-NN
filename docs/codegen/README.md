# nanoc_nn.codegen

CMSIS-NN 代码生成模块。

该模块负责读取 `nanoc_nn.converter` 生成的 `model_graph.json`、权重资料和量化信息，生成面向 Arm CMSIS-NN 的 C 推理工程。

限定场景：本模块主要面向 STM32、GD32 等 Arm Cortex-M 嵌入式低算力平台，用于在裸机或 RTOS 固件中部署小型神经网络。生成代码优先考虑固定输入、int8 量化、静态内存、有限 SRAM/Flash 和交叉编译，而不是桌面端、Linux 或高性能推理服务器。

上游约束：codegen 的唯一上游必须是 `nanoc_nn.converter`。本模块只接受 converter 的标准输出目录，不直接接受原始 ONNX 文件；缺少字段时应推动 converter schema 升级，而不是在 codegen 内部重新解析 ONNX。

## 模块定位

本模块不再自研完整神经网络算子库，而是以 CMSIS-NN 为底层内核依赖，重点实现以下能力：

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
- 当前目录是完整官方源码快照，不包含嵌套 `.git` 历史。
- 本项目代码不得手工修改、裁剪或重排 CMSIS-NN 内容；需要升级时应整体替换官方快照，或改用 submodule / 外部依赖。
- CMSIS-NN 编译还需要 CMSIS-Core 路径，后续 codegen 命令会显式要求 `--cmsis-path`。
- CMSIS-NN 主路径是 TensorFlow Lite Micro 风格的 int8/int16 量化推理；float32/float16 API 属于实验路径。

## 目录结构

```text
NanoC-NN/
├── docs/
│   └── codegen/
│       ├── README.md
│       └── plan.md
├── src/
│   └── nanoc_nn/
│       └── codegen/
│           ├── __init__.py
│           ├── __main__.py
│           ├── cli.py
│           ├── generator.py
│           ├── loader.py
│           ├── mapper.py
│           ├── memory.py
│           ├── model.py
│           ├── quantization.py
│           ├── renderer.py
│           ├── report.py
│           ├── schema.py
│           └── templates/
├── tests/
│   └── codegen/
├── third_party/
│   ├── README.md
│   └── CMSIS-NN/
└── pyproject.toml
```

## 命令入口

当前提供如下入口：

```bash
python -m nanoc_nn.codegen \
  --input build/export \
  --out build/generated \
  --cmsis-nn-root /path/to/CMSIS-NN \
  --cmsis-path /path/to/CMSIS \
  --backend mve \
  --target cortex-m4 \
  --sram-budget 128K \
  --flash-budget 512K
```

当前已经实现 converter 输出目录到 CMSIS-NN C 工程的生成闭环。float32
模型或缺少量化 section 的模型仍会报告为 `blocked`；带 Q/DQ、per-tensor
量化、`Gemm(transB=1)` 的最小 Fully Connected 路径可以生成真实
`arm_fully_connected_s8()` 调用、int8 权重和 int32 bias。Conv/Add/Pool
等其它量化算子的真实渲染仍按 blocked 处理，避免出现“映射成功但代码为空”的假象。

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
- 全连接第一版生成 `arm_fully_connected_s8` per-tensor 路径。
- FC s8 生成代码会调用 `arm_fully_connected_s8_get_buffer_size()` 校验 scratch
  buffer；其它算子的临时 buffer 仍是保守估算。
- BatchNormalization 初期只支持生成期折叠，不能折叠时阻塞生成。
- 生成报告必须给出 SRAM/Flash 估算，帮助判断能否部署到目标 STM32/GD32 型号。
- 生成的 `main.c` 仅用于 smoke test；真实固件中用户应调用 `model.h` 暴露的推理接口。
- 不生成完整 STM32CubeMX、Keil 或 GD32 IDE 工程，初期只提供纯 C 源码、CMake 示例和接入说明。

## 计划文档

技术路线和里程碑见 `docs/codegen/plan.md`。

## 本地验证

### 已有 ONNX 的一键流程

如果你已经有一个 ONNX 文件，例如：

```text
/path/to/model.onnx
```

推荐先从仓库根目录安装 editable 版本：

```bash
cd /Users/tiedan/Desktop/NanoC-NN

conda run -n nanoc-onnx-examples python -m pip install -e ".[dev]"
```

然后使用统一入口：

```bash
conda run -n nanoc-onnx-examples nanoc onnx-to-cmsis \
  --model /path/to/model.onnx \
  --target cortex-m4 \
  --sram-budget 128K \
  --flash-budget 512K
```

也可以继续使用仓库根目录下的兼容脚本：

```bash
conda run -n nanoc-onnx-examples python tools/onnx_to_cmsis_pipeline.py \
  --model /path/to/model.onnx \
  --target cortex-m4 \
  --sram-budget 128K \
  --flash-budget 512K
```

默认会在 ONNX 所在目录生成：

```text
model-nanoc-cmsis/
├── converter-output/
│   ├── model_graph.json
│   ├── model_summary.md
│   ├── conversion_report.txt
│   └── weights.h
├── cmsis-codegen/
│   ├── include/
│   ├── src/
│   ├── reports/
│   └── CMakeLists.txt
└── pipeline_report.md
```

其中 `converter-output/` 是 converter 的解析结果，`cmsis-codegen/` 是基于这些解析结果生成的 CMSIS-NN C 工程骨架。

如果希望指定输出目录：

```bash
conda run -n nanoc-onnx-examples python tools/onnx_to_cmsis_pipeline.py \
  --model /path/to/model.onnx \
  --out-root /path/to/model_codegen_result \
  --target cortex-m4
```

验证重点看：

```text
model-nanoc-cmsis/pipeline_report.md
model-nanoc-cmsis/cmsis-codegen/reports/codegen_report.txt
model-nanoc-cmsis/cmsis-codegen/reports/op_mapping.md
model-nanoc-cmsis/cmsis-codegen/reports/memory_plan.md
model-nanoc-cmsis/cmsis-codegen/reports/quantization.md
```

脚本默认会尝试使用本机 `cc` 做一次 C99 smoke compile，结果写入 `pipeline_report.md`。

### 批量本地验证

运行多模型闭环验证：

```bash
conda run -n nanoc-onnx-examples python tools/validate_local_models.py
```

该脚本会生成多种 ONNX 模型，依次执行：

```text
ONNX -> nanoc_nn.converter 输出目录 -> nanoc_nn.codegen -> C99 smoke compile
```

验证报告输出到：

```text
build/local-validation/coverage_report.md
```

单模型 codegen 示例：

```bash
python -m nanoc_nn.codegen \
  --input build/local-validation/exports/gemm_relu \
  --out build/manual-cli/gemm_relu \
  --cmsis-nn-root third_party/CMSIS-NN \
  --target cortex-m4 \
  --sram-budget 128K \
  --flash-budget 512K \
  --verbose
```
