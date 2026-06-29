# NanoC-NN-ONNX-Converter 技术路线与里程碑

## 1. 子项目目标

NanoC-NN-ONNX-Converter 是 NanoC-NN 工具链中的模型解析与转换模块，目标是将 ONNX 模型中的网络结构和权重参数提取出来，转换为便于人工编写 C 推理代码的中间资料。

本子项目不负责自动生成完整的 `model.c`，而是输出清晰、可靠、可校验的模型结构说明和 C 语言权重头文件，为后续人工拼接 C 网络结构提供依据。

## 2. 总体技术路线

整体采用 Python 实现，基于 ONNX 官方 Python API 读取模型文件，完成图结构解析、参数提取、权重导出和结果文件生成。

推荐技术栈：

- Python 3.10+
- `onnx`：读取和解析 ONNX 模型
- `numpy`：处理权重张量与数组格式转换
- `argparse`：实现命令行入口
- `pytest`：编写基础测试用例
- `torch`：仅用于生成和导出最小测试 ONNX 模型，不作为转换工具运行时依赖

核心流程如下：

```text
输入 ONNX 模型
    ↓
加载并检查模型合法性
    ↓
解析 graph、node、initializer、input、output
    ↓
提取算子类型、输入输出、参数和张量形状
    ↓
导出模型结构清单
    ↓
导出权重 C 头文件
    ↓
生成转换报告与验证辅助信息
```

### 2.1 初期支持边界

根据风险评估，初期版本需要先锁定输入模型边界，避免动态维度、数据类型和 opset 差异过早传递到 C 端。

初期约束如下：

- ONNX opset：优先支持 opset 11 到 17。超出范围时允许继续尝试解析，但必须在报告中给出警告。
- 数据类型：权重初期仅支持 `float32`。遇到 `float16`、`int8` 或其他类型时，先报告为不支持，不做隐式转换。
- 张量布局：初期默认并只验证 `NCHW`。`--layout` 参数仅用于文档标注和一致性检查，不对权重或激活张量做自动 transpose。
- 模型形态：优先面向固定输入尺寸 CNN。动态 shape 模型不作为第一阶段目标。
- 执行顺序：默认 ONNX `graph.node` 已按拓扑序排列；如果后续遇到非拓扑序模型，再补充显式拓扑排序。

## 3. 主要实现方式

### 3.1 命令行入口

提供统一命令行入口，例如：

```bash
python -m nanoc_onnx_converter --model model.onnx --out build/export
```

计划支持的参数：

- `--model`：输入 ONNX 模型路径。
- `--out`：输出目录。
- `--layout`：指定目标张量布局标注，初期默认 `NCHW`，不做实际内存重排。
- `--prefix`：生成 C 符号时使用的名称前缀。
- `--batch-size`：动态 batch 维度的固定值，初期默认 `1`。
- `--strict`：启用严格模式，遇到不支持的数据类型、动态非 batch 维度或关键 shape 缺失时直接失败。
- `--verbose`：输出更详细的解析日志。

### 3.2 ONNX 模型加载与校验

使用 `onnx.load()` 加载模型，并使用 `onnx.checker.check_model()` 做基础合法性检查。

需要记录的信息包括：

- 模型输入名称、维度和数据类型。
- 模型输出名称、维度和数据类型。
- 计算图节点顺序。
- initializer 中保存的权重张量。
- opset 版本。
- 是否存在动态维度。
- 是否存在非 `float32` 权重。

### 3.3 图结构解析

遍历 `graph.node`，为每一个节点生成结构化描述。

每层至少提取：

- 层序号。
- ONNX 节点名称。
- 算子类型，如 `Conv`、`Relu`、`Gemm`、`MaxPool`、`Softmax`。
- 输入张量名称。
- 输出张量名称。
- 属性参数，如 `kernel_shape`、`strides`、`pads`、`dilations`、`group`。
- 推断得到的输入输出 shape。

ONNX 属性使用 `AttributeProto` 表示，类型可能是 INT、FLOAT、STRING、INTS、FLOATS、TENSOR 等。实现时需要提供通用属性提取器，将属性统一转换为 Python 基础类型，并对关键算子做属性归一化。

重点归一化规则：

- `Conv`：统一 `kernel_shape`、`strides`、`pads`、`dilations`、`group`，未显式提供时填入 ONNX 默认值。
- `MaxPool`：统一 `kernel_shape`、`strides`、`pads`。
- `Gemm`：显式记录 `alpha`、`beta`、`transA`、`transB`，避免不同导出框架默认值差异造成误读。
- `Flatten`：显式记录 `axis`。

初期优先支持 NanoC-NN 章程中明确需要的基础算子：

- `Conv`
- `Relu`
- `Gemm` / `MatMul`
- `MaxPool`
- `Softmax`
- `Flatten`
- `Reshape`

后续扩展阶段优先补齐真实模型中高频出现、且对结构导出价值较高的算子：

- `Add`：常见于 bias、残差连接和导出器拆分后的逐元素加法。
- `Constant`：常用于 shape、轴、reshape 参数等小型常量张量。
- `Transpose`：常用于布局变换或导出器中间适配。
- `Cast`：常用于索引、shape 或数据类型桥接。
- `BatchNormalization`：CNN 模型中常见的归一化层。
- `GlobalAveragePool`：分类模型尾部常见的全局池化层。

这些算子的第一阶段目标仍是“结构和参数可读”，不是自动生成完整 C 推理代码。转换器需要把算子状态标为 supported，提取关键属性、输入输出 shape、相关权重/常量，并在报告中保留人工实现 C 端所需的信息。

### 3.4 Shape 推断

优先使用 ONNX 自带的 `onnx.shape_inference.infer_shapes()` 补全张量维度信息。

如果推断失败，则保留原始张量名称，并在结构清单中标注 `unknown`，避免生成误导性的维度信息。

动态维度处理策略：

- 如果动态维度位于 batch 维度，默认使用 `--batch-size` 固定为 `1`，并在 `model_summary.md` 和 `model_graph.json` 中记录原始动态标记。
- 如果动态维度不是 batch 维度，则默认报错，因为 C 端无法确定静态缓冲区大小。
- 如果未启用 `--strict`，非 batch 动态维度可以写入报告并继续导出结构清单，但不得生成声称可直接编译使用的完整权重/shape 结果。
- 任何 `-1`、字符串维度或未知维度都不得静默传递到 C 端数组尺寸宏中。

### 3.5 权重导出

从 `graph.initializer` 中读取权重张量，并转换为 C 语言头文件。

计划生成文件：

- `weights.h`：权重数组声明和维度宏定义。
- `weights.c`：可选，存放实际权重数组定义，避免头文件过大。

初期可以先生成单一 `weights.h`，等模型变大后再拆分 `.h` 和 `.c`。

权重导出约束：

- 初期仅导出 `float32` 权重。
- 非 `float32` 权重必须在报告中标注为不支持；严格模式下直接失败。
- 权重数组长度必须与 ONNX initializer 元素数量一致。
- shape 宏中不得包含动态维度或未知维度。

C 符号命名规则：

- 原始 ONNX 名称只能作为注释或映射信息，不直接作为 C 标识符。
- 将非字母、数字、下划线字符统一替换为 `_`。
- 如果清洗后名称以数字开头，则添加 `_` 前缀。
- 将连续多个 `_` 压缩为单个 `_`。
- 根据 `--prefix` 增加统一前缀，降低与用户代码冲突的概率。
- 清洗后必须进行唯一性检查；如发生冲突，追加递增序号，并在报告中记录原始名称到 C 名称的映射。

生成内容示例：

```c
#define CONV1_WEIGHT_DIMS 4
#define CONV1_WEIGHT_SHAPE {8, 1, 3, 3}

static const float conv1_weight[] = {
    0.0123f, -0.0456f
};
```

### 3.6 结构清单导出

输出人类可读的模型结构文档，作为人工编写 `model.c` 的主要依据。

计划生成文件：

- `model_summary.md`：Markdown 格式结构清单。
- `model_graph.json`：机器可读的结构化图信息，便于后续扩展。

`model_summary.md` 应包含：

- 模型输入输出。
- 每一层的算子类型、输入输出、参数、shape。
- 每一层关联的权重名称和维度。
- 当前工具支持状态，如 `supported`、`partial`、`unsupported`。

### 3.7 错误处理与兼容策略

转换工具遇到未知算子时不应直接静默忽略，而应：

- 在终端输出明确警告。
- 在 `model_summary.md` 中标注该层未支持。
- 在最终报告中列出未支持算子清单。

对于人工拼接风险较高的情况，例如 shape 缺失、权重维度异常、layout 不明确，应将问题写入转换报告。

必须重点捕获的风险：

- Shape 推断不完整。
- 动态维度无法静态化。
- C 符号命名冲突。
- 权重数据类型不支持。
- opset 版本超出初期支持范围。
- 算子属性存在框架差异，需要归一化后再导出。

### 3.8 测试与验证

测试分为三类：

- 单元测试：验证属性解析、权重命名、C 数组格式化等基础函数。
- 样例模型测试：使用小型 ONNX 模型验证完整转换流程。
- 对齐辅助测试：导出中间信息，辅助后续 C 端与 Python 端输出对比。

初期优先在 M0 阶段准备一个最小 CNN 样例模型，结构建议为：

```text
Conv -> Relu -> MaxPool -> Flatten -> Gemm -> Softmax
```

该模型用于贯穿后续所有里程碑，避免工具开发完成后才发现缺少稳定验证样例。

## 4. 建议目录结构

```text
NanoC-NN-ONNX-Converter/
├── Doc/
│   └── plan.md
├── nanoc_onnx_converter/
│   ├── __init__.py
│   ├── cli.py
│   ├── parser.py
│   ├── exporter.py
│   ├── shape.py
│   ├── naming.py
│   └── c_writer.py
├── tests/
├── pyproject.toml
└── README.md

NanoC-NN-ONNX-Examples/
├── environment.yml
├── example-1-is-over-10/
└── example-2-detect-signal-jump/
```

## 5. 里程碑清单

### M0：准备最小测试模型

目标：

- 使用 PyTorch 定义一个最小 CNN 网络。
- 网络结构覆盖 `Conv`、`Relu`、`MaxPool`、`Flatten`、`Gemm`、`Softmax`。
- 导出固定输入尺寸、固定 batch 的 ONNX 模型。

验收标准：

- `NanoC-NN-ONNX-Examples/` 中存在可复现导出脚本和生成的样例 ONNX 模型。
- 样例模型可被 `onnx.checker.check_model()` 正常校验。
- 样例模型的输入、输出 shape 明确，不依赖动态非 batch 维度。

### M1：项目骨架与命令行入口

目标：

- 建立 Python 包目录结构。
- 实现 `--model` 和 `--out` 参数。
- 实现 `--layout`、`--prefix`、`--batch-size`、`--strict` 的参数占位和基础校验。
- 能加载 ONNX 文件并输出基础模型信息。

验收标准：

- 命令行工具可以正常运行。
- 输入不存在或非法模型时能给出明确错误信息。
- 能读取并报告模型 opset、输入输出、initializer 数量和数据类型概况。

### M2：基础图结构解析

目标：

- 遍历 ONNX graph 节点。
- 提取节点名称、算子类型、输入输出张量。
- 提取常见算子属性。
- 实现通用 `AttributeProto` 属性提取器。
- 对 `Conv`、`MaxPool`、`Gemm`、`Flatten` 做基础属性归一化。

验收标准：

- 能生成包含每层基础信息的 `model_summary.md`。
- 对未知算子有明确标注。
- 不同属性类型能被转换为稳定的 Python/JSON 表示。

### M3：Shape 推断与结构清单完善

目标：

- 接入 ONNX shape inference。
- 在结构清单中补充输入输出 shape。
- 输出模型输入、输出、各层张量维度信息。
- 实现动态 batch 固定为 `--batch-size` 的策略。
- 对非 batch 动态维度给出明确错误或警告。

验收标准：

- `model_summary.md` 可作为人工编写 `model.c` 的参考。
- shape 缺失时能标注 `unknown`，而不是输出错误维度。
- 任何动态维度都不会静默进入 C 数组尺寸宏。

### M4：C 命名规则与权重导出

目标：

- 实现 C 标识符清洗规则。
- 实现清洗后名称唯一性检查。
- 读取 initializer 权重。
- 将权重转换为 C 数组。
- 为每个权重生成维度宏定义。

验收标准：

- 能生成 `weights.h`。
- 生成的 C 代码可被 C99 编译器包含。
- 权重数组长度与 ONNX 原始张量元素数量一致。
- `conv1.w` 和 `conv1_w` 这类名称清洗冲突可以被检测并自动消解。
- 非 `float32` 权重会被明确报告，不会被静默错误导出。

### M5：结构化 JSON 导出

目标：

- 生成 `model_graph.json`。
- 保存节点、属性、shape、权重映射等机器可读信息。
- 保存原始 ONNX 名称与清洗后 C 符号名称的映射。
- 保存动态维度处理结果和 opset 信息。

验收标准：

- JSON 文件可被 Python 正常读取。
- 字段结构稳定，便于后续 C 端工程或验证工具复用。
- JSON 能完整复现 Markdown 结构清单中的关键技术信息。

### M6：样例模型端到端转换

目标：

- 跑通从 ONNX 输入到结构清单、权重头文件、JSON 图描述的完整流程。
- 使用 M0 准备的最小 CNN 模型作为首个固定验证基准。

验收标准：

- 样例模型转换无异常。
- 输出文件完整且内容可人工检查。
- 生成的 `weights.h` 可通过 C99 编译器语法检查。

### M7：测试与文档补齐

目标：

- 为核心解析和导出逻辑补充单元测试。
- 完善 README 中的安装、运行和输出说明。
- 记录当前支持的 ONNX 算子列表。
- 补充风险处理说明，包括动态维度、opset、数据类型、命名冲突和 layout 边界。

验收标准：

- 基础测试可通过。
- 新用户可以根据 README 完成一次样例模型转换。
- README 明确说明初期支持边界，避免误用到量化模型或动态 shape 模型。

### M8：真实模型高频算子扩展

目标：

- 支持 `Add`、`Constant`、`Transpose`、`Cast`、`BatchNormalization`、`GlobalAveragePool` 的结构解析。
- 为每个新增算子提供稳定的属性归一化结果。
- 区分参数 initializer 和辅助常量 initializer，例如 `Reshape` 的 INT64 shape 常量不应作为 C 权重导出。
- 保持未知算子仍能在非严格模式下继续导出结构报告。

验收标准：

- 新增算子不会出现在 unsupported ops 清单中。
- `model_graph.json` 能保存新增算子的关键属性。
- `weights.h` 只导出 float32 参数权重，不导出 shape、axis 等辅助常量。
- 单元测试覆盖新增算子默认属性和 initializer 分类。

### M9：开源 ONNX 模型库准备

目标：

- 在仓库根目录创建 `onnx-model/` 临时测试目录。
- 从网络上的经典开源 ONNX 模型来源下载至少 10 个 ONNX 文件。
- 优先选择覆盖 CNN、轻量分类网络、池化、BN、Add、Transpose、Cast、Constant 等常见模式的模型。
- 记录模型来源、文件名、大小和下载结果，便于后续复测。

验收标准：

- `onnx-model/` 下至少存在 10 个可被 `onnx.load()` 读取的 ONNX 文件。
- 每个模型来源可追溯。
- 下载失败的模型必须在报告中记录，不计入覆盖数量。

### M10：模型库轮询转换测试

目标：

- 对 `onnx-model/` 中的模型逐个执行 converter。
- 每个模型输出独立转换结果目录，避免互相覆盖。
- 收集节点数、initializer 数、C 权重数、unsupported ops、警告数量、转换是否成功等数据。
- 对生成的 `model_graph.json` 做 JSON 可读性检查。
- 对生成的 `weights.h` 做 C99 include 级别语法检查。

验收标准：

- 生成统一测试报告，能够横向比较每个模型的转换情况。
- 转换失败、shape 推断失败、非 float32 参数、未知算子等风险能被明确定位。
- 测试轮询结束后先停止，不在同一轮中继续修复，以便人工评估下一步优先级。

### M11：基于测试报告收敛支持边界

目标：

- 根据 M10 报告统计真实模型中仍未覆盖的算子和数据类型。
- 按出现频率、实现成本和 C 端价值拆分后续任务。
- 明确哪些问题属于 converter 应解决，哪些属于 C operators 或模型前处理阶段解决。

验收标准：

- 输出后续优先级清单。
- plan 和 README 中同步当前支持边界。
- 不把测试中的偶发现象直接扩展成无边界兼容承诺。

## 6. 初期边界

初期不实现以下能力：

- 自动生成完整 `model.c`。
- 自动图调度与内存复用规划。
- 全量 ONNX 算子兼容。
- 量化模型解析。
- 非 `float32` 权重自动转换。
- 自动 layout transpose。
- 动态输入尺寸模型的 C 端缓冲区规划。
- 针对特定 MCU 的格式优化。

这些能力可以作为后续版本扩展，但当前版本优先保证结构清晰、权重准确、输出可人工使用。
