# AGENTS.md

本文件用于说明智能体在本仓库中协作开发时应遵守的规则。

## 项目定位

本项目当前定位为 **CMSIS-NN ONNX 端侧代码生成器**，目标是从 ONNX 模型生成基于 Arm CMSIS-NN 的 Cortex-M 推理 C 工程。项目限定场景是 STM32、GD32 等 Arm 架构嵌入式低算力平台上的神经网络开发，优先服务裸机或 RTOS 固件，而不是桌面端、Linux 或 Arm A-class 推理。当前采用根级 Python 工程组织核心能力：

- `src/nanoc_nn/converter`：Python 编写的 ONNX 前端解析与中间表示导出模块。
- `src/nanoc_nn/codegen`：基于 converter 输出生成 CMSIS-NN C 推理工程的代码生成模块。
- `src/nanoc_nn/pipeline`：串联 ONNX 解析与 CMSIS-NN 代码生成的一体化编排模块。
- `NanoC-NN-ONNX-Examples`：暂时无损保留的 ONNX 教学与验证样例集合。

## 开发方法论：测试驱动开发

本项目采用测试驱动开发（TDD）作为核心开发方法。这不是可选的辅助手段，而是必须遵守的开发纪律。

**为什么**：我们在为未知输入空间构建代码生成器。我们不知道市场上会出现什么样的 ONNX 模型，因此不可能"先想清楚所有情况再写代码"。唯一可靠的方式是：通过不断构造测试模型、执行生成器、观察结果、修复缺陷，逐步扩展代码生成器的能力边界。

**核心循环**：

1. 构造一个 ONNX 测试模型（定义需求）
2. 喂给代码生成器（执行）
3. 观察结果是否符合预期（评估）
4. 不符合则修复代码生成器（迭代）
5. 符合则永久纳入回归测试集（沉淀）

**智能体必须遵守的 TDD 规则**：

- **迭代起点是测试用例，不是代码**。每次开发迭代开始时，必须先遍历已有测试用例，从中发现能力缺口（已有链路的细化雕刻、或新场景的探索），然后新增测试用例、执行测试、根据失败报告做开发。禁止跳过测试直接修改代码。
- **用户反馈 → 测试用例，不是 → 直接改代码**。当用户报告某个 ONNX 模型有问题，第一步是把问题分解成若干测试用例写进 `cases/`，执行确认失败，再按报告修复。跳过用例直接修复，等于放弃了对此次修复的永久验证手段。
- **任何功能修改或新增，必须有对应的测试用例覆盖**。不允许出现"改了代码但没有测试验证"的提交。
- **修复 bug 时，先构造能复现 bug 的测试用例，再修复代码**。测试用例是 bug 不再复发的保证。
- **不允许为了让测试通过而削弱测试**。如果测试暴露了真实问题，修复代码而不是修改测试预期。
- **全量回归是每次交付的强制门槛**。新增用例通过后，必须跑全量才能确认没有破坏已有能力。全量全通 = 新版能力集诞生，这是产品向前移动的最小完整单元。
- **"预期拒绝"同样是有效测试**。代码生成器必须能正确拒绝它不支持的输入（返回 `blocked` 或 `unsupported`），而不是静默产出错误代码或崩溃。
- **测试报告就是开发任务单**。不需要另外的需求文档——跑一遍测试，失败的就是接下来要做的事。

测试资产和详细计划位于 `tdd/` 目录，纲领性说明见 `tdd/README.md`。

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
- **生成物优先于报告状态**：不得仅凭 `codegen_report.txt`、`pipeline_report.md` 或 CLI 输出中的 `status: ok` 宣称转换成功。`ok` 交付必须检查 `cmsis-codegen/src/model.c`：`nanoc_model_run()` 中应存在真实 CMSIS-NN 调用路径，C 预处理结构完整，不得只有 `Generated execution trace`、fallback stub、孤立 `#else/#endif` 或空运行路径，并且应通过 C99 smoke compile。报告为 `ok` 但生成物不可编译或不可推理，属于严重假阳性，优先级高于普通 blocked/unsupported，必须先沉淀测试或验证规则再继续扩展能力。
- **TDD 回归与能力集保护**：任何 converter 或 codegen 改动后，必须执行 `python tdd/scripts/run_tests.py` 确认全部测试用例结果符合预期。该命令同时更新 `tdd/CAPABILITIES.md`。若已确认能力（之前 PASS 的用例）出现退化，视为严重缺陷，不得提交。新增算子支持时，必须同步新增或激活对应的 TDD 测试用例，用例通过后能力集自动扩展。

## TDD 操作流程

智能体进入 `tdd/` 目录进行开发迭代时，必须遵循以下流程：

1. **读 STATUS.md**：了解当前能力矩阵、能力集状态和下一步方向
2. **读 CAPABILITIES.md**：了解哪些能力已确认、哪些目标尚未达成
3. **遍历 cases/ 目录**：审视已有测试用例，发现能力缺口
4. **在 cases/ 中新增用例规格**：按 `cases/README.md` 规范编写
5. **在 generate_models.py 中注册生成函数**：复用 `common/model_builder.py` 构建 ONNX
6. **在 run_tests.py 的 EXPECTED_STATUS 中注册预期**
7. **执行测试**：`python tdd/scripts/run_tests.py --generate`
8. **根据测试报告修改 src/ 代码**
9. **全量回归**：`python tdd/scripts/run_tests.py`（自动更新 CAPABILITIES.md）
10. **更新 STATUS.md 和当前迭代记录**

### 能力集规则

- **CAPABILITIES.md 是产品能力的唯一权威声明**。它由 run_tests.py 自动生成，禁止手动编辑。
- **能力只能通过测试获得**。不允许在任何文档中声称支持某种能力而没有对应的 PASS 用例。
- **能力不允许退化**。全量回归中任何已确认能力（之前 PASS 的用例）变为 FAIL，必须视为严重缺陷立即修复，不得跳过。
- **CAPABILITIES.md 必须纳入 git**。每次能力集变化（新增 PASS 或修复退化）都应在提交中体现，commit history 即能力增长记录。
- **开发的终极目标是扩大能力集**。新功能 = 新的 PASS 用例，bug 修复 = 修复退化的 PASS 用例。

关键目录说明：
- `tdd/STATUS.md`：入口文件，全局状态总览
- `tdd/CAPABILITIES.md`：能力集权威声明（自动生成，git 跟踪）
- `tdd/cases/`：测试用例规格，永久资产
- `tdd/scripts/`：模型生成和测试执行脚本
- `tdd/models/`：生成的 ONNX 模型（gitignore，按需重新生成）
- `tdd/results/`：执行结果（latest.json 为当前基线）
- `tdd/iterations/`：迭代记录，开发历史积累
