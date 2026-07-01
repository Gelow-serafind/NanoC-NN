# AGENTS.md

本文件用于说明智能体在本仓库中协作开发时应遵守的规则。

## 项目定位

本项目当前定位为 **CMSIS-NN ONNX 端侧代码生成器**，目标是从 ONNX 模型生成基于 Arm CMSIS-NN 的 Cortex-M 推理 C 工程。项目限定场景是 STM32、GD32 等 Arm 架构嵌入式低算力平台上的神经网络开发，优先服务裸机或 RTOS 固件，而不是桌面端、Linux 或 Arm A-class 推理。当前采用根级 Python 工程组织核心能力：

- `src/nanoc_nn/converter`：Python 编写的 ONNX 前端解析与中间表示导出模块。
- `src/nanoc_nn/codegen`：基于 converter 输出生成 CMSIS-NN C 推理工程的代码生成模块。
- `src/nanoc_nn/pipeline`：串联 ONNX 解析与 CMSIS-NN 代码生成的一体化编排模块。
- `NanoC-NN-ONNX-Examples`：暂时无损保留的 ONNX 教学与验证样例集合。

## 通用原则

- 修改前先阅读相关 `README.md`、`docs/` 文档和已有代码。
- 保持项目边界清晰：converter 负责 ONNX 解析和稳定中间表示，codegen 负责生成 `model.c`、工程骨架和 CMSIS-NN 调用代码。
- 不引入与当前里程碑无关的大型框架或复杂抽象。
- 新增文件默认使用 UTF-8 和 LF 换行。
- 文档优先使用中文，代码中的标识符、注释和错误信息优先使用英文。
- 设计生成代码时默认面对 SRAM/Flash 有限的 MCU，不假设文件系统、堆内存、POSIX API 或操作系统服务存在。
- codegen 的唯一上游必须是 converter 标准输出；不得在 codegen 中直接解析原始 ONNX 或建立平行前端。
- `third_party/CMSIS-NN/` 是 Arm 官方开源库源码快照，不得手工修改、裁剪、格式化或重排其内容；需要升级时只能整体替换为官方新快照，或改用 submodule / 外部依赖。

## Git 操作规则

- 智能体不得随意执行最终性的 Git 操作，包括但不限于 `git commit`、`git push`、创建分支、切换分支、删除分支、变基、合并、打标签和回滚历史。
- 任何最终性的 Git 操作在执行前必须向用户说明将要执行的命令、涉及的文件或分支，并获得用户明确确认。
- 可以执行只读 Git 命令来了解状态，例如 `git status`、`git diff`、`git log`、`git branch --show-current`。
- 关于 `git push`：因网络可能不稳定，用户确认允许 push 后，应使用循环脚本持续尝试 push，直到 push 成功；智能体需要持续监控脚本输出，不得在脚本仍需监控时结束任务。
- 提交应鼓励按修改范围分批进行，优先做到一个提交对应一个清晰的改动主题。
- commit message 必须使用以下前缀之一：
  - `修改[CHG]`：已有功能、文档或配置的调整。
  - `新增[ADD]`：新增功能、文档、测试或配置。
  - `删除[DEL]`：删除文件、功能或废弃内容。
  - `修复[FIX]`：修复缺陷、错误行为或测试失败。
- commit message 应描述项目内容变化，不允许出现“agent 共同提交”、共同作者、AI 生成签名或类似归因内容。

## Python 模块规则

适用于 `src/nanoc_nn/converter/`：

- 目标 Python 版本为 3.10+。
- 命令行入口使用 `argparse`。
- ONNX 解析使用官方 `onnx` Python API。
- 数组处理使用 `numpy`。
- 测试使用 `pytest`。
- 初期已支持 `float32` 权重、固定输入尺寸 CNN、默认 `NCHW` 布局；后续需要补充量化参数提取能力，为 CMSIS-NN codegen 服务。
- C 符号生成必须经过命名清洗和唯一性检查。
- 动态 batch 可以按配置固定为 1，非 batch 动态维度不得静默传递到 C 端。

适用于 `NanoC-NN-ONNX-Examples/`：

- 样例代码可以依赖 PyTorch 和 ONNX Runtime，但不得成为 converter 运行时依赖。
- 每个样例的网络定义必须单独放在 `scripts/network.py`。
- 样例生成物写入 `outputs/`，不纳入版本管理。
- 样例 README 需要说明训练、checkpoint 推理、ONNX 推理和参数传入方式。

## CMSIS-NN Codegen 模块规则

适用于 `src/nanoc_nn/codegen/`：

- codegen 读取 converter 产物，优先以 `model_graph.json` 作为中间表示输入。
- codegen 不接受原始 ONNX 作为直接输入；若生成需要更多字段，应升级 converter schema，而不是绕过 converter。
- 生成的 C 代码优先兼容 C99，并以 CMSIS-NN/CMSIS-Core 作为目标依赖。
- 不再自研完整神经网络算子库，优先映射到 CMSIS-NN 已提供的优化内核。
- 生成代码不得在运行期调用 `malloc` 或 `free`；激活缓冲区、临时 buffer 和权重布局由生成阶段规划。
- 必须在报告中记录 ONNX 算子到 CMSIS-NN API 的映射关系、量化参数、buffer 大小和不支持原因。
- 张量布局转换必须显式记录，不得静默假设 NCHW/NHWC 互通。
- 针对 CMSIS-NN API 版本的约束需要写入 README 或生成报告。
- codegen 目标平台优先为 STM32、GD32 等 Cortex-M MCU；生成物应易于接入 STM32CubeIDE、Keil MDK、Arm GCC/CMake 或厂商固件工程。
- 生成报告必须尽量包含目标内核、backend、SRAM/Flash 预算和超预算风险。
- 不直接绑定具体 HAL、启动文件、链接脚本或 IDE 工程格式，除非后续里程碑明确要求。

## 验证要求

- 文档类修改至少检查 Markdown 结构和链接路径。
- Python 代码修改后优先运行 `pytest`。
- 生成的 C 头文件需要能被 C99 编译器包含。
- codegen 生成物后续需要优先用 Arm GNU Toolchain/Arm Compiler/FVP 或目标开发板验证，PC 端检查只能作为早期 smoke test。
- 对高风险逻辑补测试，包括 shape 处理、权重命名、属性解析和 C 数组导出。
