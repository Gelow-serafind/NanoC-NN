# NanoC-NN int8 ONNX 到 CMSIS-NN C 代码生成单链路计划

## 1. 当前项目目标

项目当前只收敛一条链路：

```text
任意 int8 ONNX 输入
    ↓
converter 解析 ONNX
    ↓
检查是否缺失算子、shape、layout、量化字段
    ↓
缺失则终止并输出明确错误报告
    ↓
转换成功后调用 codegen
    ↓
基于 CMSIS-NN 生成 int8 量化 C 代码
```

这里的“任意 int8 ONNX”不是“任意 ONNX 算子都能生成”，而是指用户可以给工具一个
ONNX 文件；工具必须自己判断它是否满足当前支持边界。满足则生成代码，不满足则明确失败，
不能生成看似成功但无法推理的工程。

## 2. 不再展开的方向

为了让工程先形成可交付闭环，以下内容暂时不作为当前阶段目标：

- float32/f16 CMSIS-NN codegen。
- 自动训练、PTQ、QAT 或隐式 float32 到 int8 量化。
- 全量 ONNX 算子兼容。
- 自研 C 神经网络算子库。
- 直接生成完整 STM32CubeMX、Keil 或 GD32 IDE 工程。
- 面向 Linux、桌面端、Arm A-class 或其它非 Cortex-M 后端。
- 真实模型大池子覆盖率扩张。

CMSIS-NN 官方确实存在 f32/f16 API，但本项目当前只做 int8 主链路。这是项目边界，
不是对 CMSIS-NN 官方能力的否定。

## 3. 代码走读后的当前状态

### 3.1 已经存在

- `src/nanoc_nn/pipeline` 已经提供一键入口，可以从 ONNX 路径生成：
  - `converter-output/`
  - `cmsis-codegen/`
  - `pipeline_report.md`
- `src/nanoc_nn/converter` 已能解析 ONNX、导出 `model_graph.json`、结构报告和
  `weights.h`。
- converter 已能识别第一版 Q/DQ 量化信息。
- 对 Q/DQ、per-tensor、`Gemm(transB=1)` 的 Fully Connected 模型，converter
  已能输出 `arm_fully_connected_s8()` 所需字段。
- `src/nanoc_nn/codegen` 已能读取 converter 输出目录。
- codegen 已能为 Fully Connected s8 路径生成真实 CMSIS-NN C 调用、int8 权重和
  int32 bias。
- pipeline 已能做 C99 smoke compile，用于早期语法检查。

### 3.2 当前不足

- 现在真实可运行 C 代码覆盖 `Gemm` / `MatMul` 的 Fully Connected 第一版路径，以及
  普通 `Conv(group=1, dilation=1)` 的 `arm_convolve_wrapper_s8()` 第一版路径。
- DepthwiseConv、Add、Pool、Softmax、Transpose 等仍主要停留在 mapping 或 blocked
  报告，尚未生成真实 CMSIS-NN 调用。
- converter 的 Q/DQ 提取当前偏向 FC per-tensor 路径，还没有覆盖 Conv per-channel、
  Add/Pool/Softmax 等算子所需完整量化参数。
- pipeline 默认会继续生成 blocked 工程骨架；后续应区分“调试模式允许骨架”和
  “交付模式遇到 blocked 直接失败”。
- 部分测试仍使用 float ONNX 验证 pipeline 输出目录，这类测试应保留为负向或兼容测试，
  不应作为主链路验收标准。
- 工程中存在本地缓存、构建产物、样例输出等非项目文件，应持续清理并依赖 `.gitignore`
  控制。

## 4. 目标使用方式

最终用户只需要给出 ONNX 路径：

```bash
nanoc onnx-to-cmsis \
  --model /path/to/model.int8.onnx \
  --target cortex-m4 \
  --sram-budget 128K \
  --flash-budget 512K
```

默认输出到 ONNX 所在目录：

```text
model.int8-nanoc-cmsis/
├── converter-output/
├── cmsis-codegen/
└── pipeline_report.md
```

成功条件：

- 输入模型是 Q/DQ int8 ONNX。
- shape 固定，或只有 batch 可固定为 1。
- 所有运行期算子都有 converter schema 支持。
- 所有运行期算子都有 codegen renderer 支持。
- 所有运行期算子具备 CMSIS-NN s8 所需量化参数。
- 生成报告无 blocked / unsupported。

失败条件：

- 非 int8 Q/DQ 模型。
- 缺量化字段。
- 动态 shape 无法固定。
- 出现未支持 ONNX 算子。
- 出现已识别但尚未实现 CMSIS-NN renderer 的算子。
- layout 无法安全确认或转换。
- SRAM/Flash 超过用户预算。

失败时必须输出明确错误和报告，不生成或不宣称生成可推理 C 工程。

## 5. 里程碑

### M0：工程清理与文档收敛

目标：

- 删除缓存、临时输出和与当前链路无关的历史文件。
- 保留 `NanoC-NN-ONNX-Examples`，但不把它作为当前主工程结构的一部分继续扩张。
- 重写 roadmap 和 codegen plan，使其只服务 int8 ONNX 到 CMSIS-NN C 代码生成链路。

验收：

- `git status` 中只出现真实源码/文档改动。
- 文档不再把 float codegen、真实模型大覆盖或多后端作为当前主线。

### M1：交付模式的失败语义

状态：已完成第一版。

目标：

- pipeline 默认面向交付链路：出现 blocked / unsupported 时返回非零退出码。
- 增加显式调试选项，例如 `--allow-blocked-output`，用于保留骨架和报告。
- pipeline 报告中区分：
  - `ok`：生成物包含真实 CMSIS-NN int8 推理代码。
  - `blocked`：当前缺字段或缺 renderer。
  - `unsupported`：当前不支持该算子或模型形态。

验收：

- 非 Q/DQ float ONNX 默认返回非 0，并保留 converter/codegen 报告。
- 当前支持的 Q/DQ FC 模型默认成功。
- blocked 模型默认按交付失败处理；仅在显式 `--allow-blocked-output` 下允许把
  blocked 骨架视为调试成功输出。

### M2：converter int8 输入契约固化

状态：进行中，已加入 `model_graph.json.quantization.int8_contract` 第一版。

目标：

- 明确 `model_graph.json` 的 int8 codegen 必需字段。
- 将 Q/DQ tensor、weight、node 量化信息整理为稳定 schema。
- 对每个运行期节点输出：
  - 输入/输出 tensor 的 scale 和 zero_point。
  - 权重量化数据。
  - bias int32 数据或可生成依据。
  - CMSIS-NN 所需 multiplier / shift / offset / activation range。
  - layout 信息。

验收：

- schema 文档和测试覆盖 FC 当前路径。
- 缺字段时 converter 或 codegen 能定位到具体节点和字段。

### M3：Fully Connected 链路产品化

目标：

- 把当前 Q/DQ `Gemm(transB=1)` -> `arm_fully_connected_s8()` 代码生成做成稳定基线。
- 完善 MatMul/Gemm 支持边界，不支持的形式必须失败。
- 补 C 文件、权重文件、报告和测试。

验收：

- 一个单层或多层 FC int8 Q/DQ ONNX 可以一键生成 C 工程。
- 生成的 `model.c` 包含真实 `arm_fully_connected_s8()` 调用。
- C99 smoke compile 通过。

### M4：Conv / DepthwiseConv int8 支持

状态：普通 `Conv(group=1, dilation=1)` 已完成第一版真实 renderer；DepthwiseConv
仍进行中。

目标：

- converter 提取 Conv / DepthwiseConv 所需 Q/DQ 参数。
- 处理 ONNX 权重布局到 CMSIS-NN 所需布局。
- codegen 生成 `arm_convolve_wrapper_s8()` 和
  `arm_depthwise_conv_wrapper_s8()` 调用。
- 使用 CMSIS-NN buffer size getter 或保守静态规划记录 scratch。

验收：

- 一个 Conv -> Relu -> FC 的 Q/DQ int8 ONNX 可以生成真实 C 调用。
- layout 转换在报告中明确可追溯。
- 不支持的 padding、dilation、group 或 per-channel 形态明确失败。

### M5：Pooling / Add / Softmax int8 支持

状态：Pooling 和 Softmax 第一版已完成；Add 仍 blocked。

目标：

- 支持 MaxPool、AveragePool、GlobalAveragePool。
- 支持 Add 常见残差或 bias-like 逐元素加法。
- 支持 Softmax 输出分类结果。
- 补齐各自 CMSIS-NN 参数和报告。

验收：

- 小型 CNN 分类模型可以完整生成 int8 C 推理代码。
- 任何尚未实现参数计算的算子不得被标记为 ok。

### M6：一键 CLI 验收

目标：

- 固化最终用户命令。
- 默认在 ONNX 所在目录生成结果目录。
- 成功时输出可移植的 C99 源文件和报告。
- 失败时返回非零，并指向具体报告文件。

验收：

- `nanoc onnx-to-cmsis --model xxx.int8.onnx` 是主入口。
- 所有主链路测试都使用 int8 Q/DQ ONNX。
- README 给出成功、失败和查看报告方式。

### M7：嵌入式接入最小验证

状态：生成工程已输出 `firmware_integration.md` 第一版；Arm 工具链或开发板验证仍待执行。

目标：

- 生成代码保持纯 C99，不依赖 HAL、文件系统、线程或堆内存。
- 输出 STM32/GD32 类工程接入说明。
- 至少用 Arm 工具链或等价交叉编译方式验证 include/source 组织。

验收：

- 用户能将 `model.c`、`model.h`、`model_weights.h` 接入固件工程。
- 报告说明 CMSIS-NN/CMSIS-Core include 和编译宏要求。

## 6. 下一步开发顺序

1. 完成 M0：清理缓存和重写计划。
2. 修改 pipeline 默认失败语义，新增调试模式。
3. 把现有 float pipeline 测试改为负向测试。
4. 强化 Q/DQ FC 测试，作为第一条成功链路。
5. 按 Conv、Pool/Add/Softmax 的顺序补 converter schema 和 codegen renderer。
