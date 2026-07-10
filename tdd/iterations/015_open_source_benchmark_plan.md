# 迭代 015: 开源对标与下一阶段计划

## 基本信息

- **日期**: 2026-07-11
- **前置迭代**: 014
- **触发来源**: 人类需求 / 阶段性技术路线评估

## 背景

本轮先将两个相近开源项目拉取到临时研究目录：

- `tdd/work/comparative_research/nnom`
- `tdd/work/comparative_research/onnx2c`

这两个目录仅作为临时对标材料，不纳入正式 fixture、release 包或能力集声明。

对标目标不是寻找可直接替换 NanoC-NN 的方案，而是识别可借鉴的工程组织、测试体系、内存规划和嵌入式落地经验，为下一阶段 TDD 迭代制定计划。

## 两个开源工程可借鉴点

### NNoM

NNoM 更接近 MCU 神经网络运行时库，而不是 ONNX codegen 编译器。它的重点是把模型以 layer/runtime 的形式部署到微控制器，并提供本地 C backend 与 CMSIS-NN backend 选择。

可借鉴点：

- **嵌入式 layer 生命周期**：每个 layer 拥有 build/run/free 等职责拆分。NanoC-NN 后续可借鉴这种边界，把当前集中在 `generator.py` 的 Conv、Gemm、Pool、Concat、Softmax 等生成逻辑拆成独立 emitter。
- **内存 block 与静态 buffer 思路**：NNoM 有 block 复用、静态 buffer、运行时 memory stat 等概念。NanoC-NN 应在生成期完成类似的 tensor lifetime 分析和 buffer 复用，而不是只保守保留两块最大 activation buffer。
- **port 配置层**：NNoM 使用 `port/` 层抽象 malloc/free、日志、计时、CMSIS-NN 开关等平台差异。NanoC-NN 后续可建立 Cortex-M3/M4/M7/M55 profile，将 SRAM/Flash 预算、FPU/DSP/MVE、CMSIS-NN API 可用性放入显式平台配置。
- **真实边缘场景样例**：NNoM examples 包含 keyword spotting、MNIST、HAR、RNN denoise 等 MCU 场景。它们非常适合反向启发 NanoC-NN 的真实网络 fixture 和数据集建设。
- **量化与权重预处理经验**：NNoM 脚本中包含 BN fuse、per-channel quant、dense weight reorder 等模型部署前处理。NanoC-NN 不应照搬 Keras workflow，但可以借鉴这些处理目标，转化为 ONNX/QDQ 图上的 pass 或测试用例。

不适合直接采用的部分：

- NNoM 不是以任意 ONNX 作为第一入口。
- NNoM 依赖自己的 runtime layer 框架，而 NanoC-NN 的目标是生成可直接接入固件工程的 CMSIS-NN C 项目。
- NNoM 运行时内存模型和 NanoC-NN 的“生成期规划、运行期无 malloc/free”原则不同。

### onnx2c

onnx2c 是 ONNX 到 C 的编译器，和 NanoC-NN 架构最接近。它的目标同样包含 TinyML 和微控制器，但生成的是通用 C 代码，重点不是 CMSIS-NN int8 kernel 映射。

可借鉴点：

- **Graph/Tensor/Node 编译器结构**：onnx2c 将 ONNX graph 解析为 graph、tensor、node 对象，每个 ONNX op 拥有独立 node 实现。NanoC-NN 应把 codegen 重构为类似结构，减少 `generator.py` 的持续膨胀。
- **每 op 一个 emitter**：onnx2c 的 `src/nodes/` 目录将 Conv、Gemm、Pool、Reshape、QLinearConv 等逻辑分开。NanoC-NN 后续新增算子时，也应先新增最小测试，再新增对应 emitter。
- **optimization pass 管线**：onnx2c 已有 unionize tensor、fold casts 等优化 pass。NanoC-NN 应建立 pass pipeline，用于 QDQ fold、Relu/Clip fuse、layout normalize、memory reuse、platform budget check。
- **tensor union 内存复用**：onnx2c 根据消费者是否已执行来释放 tensor union slot。NanoC-NN 可以借鉴这个生命周期模型，生成更贴近 MCU SRAM 约束的 activation plan。
- **ONNX backend tests 思路**：onnx2c 的测试体系大量复用 ONNX 官方 backend node tests。NanoC-NN 可以从中选择 int8/QDQ 和静态 shape 相关用例，转化为 TDD core case。
- **节点乱序解析与诊断**：onnx2c 能多轮解析尚未 ready 的 node。NanoC-NN 后续在 converter 层也应加强节点依赖、未解析输入、动态 shape 和 unsupported reason 的诊断质量。

不适合直接采用的部分：

- onnx2c 是 generic C / float-first，不是 CMSIS-NN int8-first。
- onnx2c 不负责 Arm CMSIS-NN API 参数映射、scratch getter、activation min/max、per-channel multiplier/shift 等细节。
- onnx2c 没有 NanoC-NN 当前需要的 `ok / blocked / unsupported / oversize` 产品语义。
- onnx2c 的目标不是生成 STM32/GD32 固件接入包和平台预算报告。

## 对 NanoC-NN 当前路线的判断

当前 NanoC-NN 路线不需要推倒。它相对两个对标项目的核心优势是：

- 以 int8/QDQ ONNX 到 CMSIS-NN C 工程为明确产品目标。
- converter、codegen、pipeline 分层已经建立。
- TDD 已经形成固定循环，能力由测试获得，不靠文档声称。
- 已完成 MNIST int8 与 signal jump int8 的 ONNX-vs-C 数值闭环。
- 已具备真实网络导入、正确拒绝、平台预算 `oversize` 等产品语义。

当前主要风险是：

- `src/nanoc_nn/codegen/generator.py` 已承载过多职责，继续扩算子会降低可维护性。
- float reference C 是重要验证工具，但不等同于 CMSIS-NN int8 正式交付能力，需要在文档和能力集展示中继续区分。
- activation buffer 规划仍偏保守，没有完整 tensor lifetime 分析。
- 真实外部 QDQ 网络的量化/布局组合仍有限，SqueezeNet 暴露的 QDQ + Concat + GlobalAveragePool + Softmax 仍是重要缺口。
- 缺少逐层 dump 工具，完整网络失败时定位成本较高。

## 下一阶段路线调整

下一阶段最重要的变化不是先扩更多算子，而是调整 TDD 开发架构：以 ONNX 官方 schema/opset 为基线，以表驱动的支持矩阵作为中间层，再用 TDD 验证每一个支持子形态。

新的开发心智：

```text
ONNX schema/opset 是前端真相
NanoC-NN support matrix 是产品承诺
CMSIS-NN lowering 是后端实现
TDD case 是能力获得方式
```

这意味着：

- 不再从业务名称直接发明前端算子，例如 `Conv1d`、`Conv2d`、`DepthwiseConv`。
- 应先承认 ONNX 中它们都是 `Conv`，再根据 rank、group、kernel_shape、strides、dilations、pads、auto_pad、量化形态等字段进入不同 lowering 分支。
- TDD 用例命名可以保留业务可读性，但用例规格必须说明它覆盖的是哪个 ONNX op schema 的哪个子形态。
- 新增算子支持时，第一步不是改 renderer，而是补 support matrix 行和最小 TDD case。
- 真实网络暴露的问题要反向拆成 schema 子形态，而不是直接堆完整网络特判。

建议建立两张表：

### ONNX schema 基线表

用于记录 ONNX 官方层面的事实：

| 字段 | 含义 |
|------|------|
| `domain` | ONNX domain，通常为空或 `ai.onnx` |
| `op_type` | 官方算子名，例如 `Conv`、`Gemm`、`MaxPool` |
| `since_version` | schema 版本 |
| `inputs` / `outputs` | 官方输入输出约束 |
| `attributes` | 官方属性和默认值 |
| `type_constraints` | 官方类型约束 |
| `shape_rules` | 形状推导和 rank 约束 |

### NanoC-NN support matrix

用于记录 NanoC-NN 产品层面的承诺：

| 字段 | 含义 |
|------|------|
| `op_type` | 对应 ONNX 官方算子 |
| `opset_range` | 当前策略覆盖的 opset 范围 |
| `form` | 支持的子形态，例如 `rank=4, group=1, QDQ int8` |
| `lowering` | lowering 类型，例如 `cmsis_conv2d_s8` |
| `cmsis_api` | 对应 CMSIS-NN API |
| `status` | `ok` / `blocked` / `unsupported` / `oversize` |
| `required_fields` | 进入该分支必须具备的字段 |
| `reject_reason` | 不支持时的稳定拒绝原因 |
| `tdd_cases` | 保护该能力的最小用例和完整网络用例 |

示例：

| ONNX op | 子形态 | NanoC-NN lowering |
|---------|--------|-------------------|
| `Conv` | rank=4, group=1, QDQ/int8 | `arm_convolve_wrapper_s8` |
| `Conv` | rank=4, group=input_channels, multiplier>=1 | `arm_depthwise_conv_wrapper_s8` |
| `Conv` | rank=3, group=1, QDQ/int8 | 1D Conv lowering 到 CMSIS-NN 兼容布局 |
| `Conv` | dilation!=1 或未知动态 shape | `blocked` 或 `unsupported` |

这会把之前“通过 TDD 不断补分支”的方式，升级为“ONNX schema 基线 + support matrix 分支 + TDD 验证”的方式。TDD 仍是内核，但测试不再漂浮在业务现象上，而是绑定到标准算子语义。

### 新模型 blocked 后的标准分流

当一个新的真实 ONNX 模型进入测试并出现 `blocked`、`unsupported`、生成物不完整或数值不一致时，下一步不是直接修代码，而是先按能力地图分流：

1. 读取模型 `opset_import` 和 node 列表。
2. 对照 `tdd/onnx_schema/` 的官方 catalog，确认涉及的 ONNX op 是否属于当前绑定版本的官方 schema。
3. 对照 NanoC-NN support matrix：
   - 如果 op 不在 support matrix 中，说明是未支持算子，新增 planned/blocked support 行和最小 TDD case。
   - 如果 op 已在 support matrix 中，但模型的 rank、group、axis、dtype、opset、layout、Q/DQ 模式或动态 shape 不匹配，说明是已支持算子的未覆盖特殊分支，细化 schema 子形态并新增最小 TDD case。
   - 如果 op 不在官方 catalog 但真实模型出现，标记为 `schema_source="extension"`，不能冒充官方 ONNX 支持。
4. 先让最小 case 稳定失败，再修改 converter/codegen。
5. 最小 case 通过后回测原始完整网络，再跑全量回归。

这个流程把 TDD 从“盲目探索”改成“沿着 ONNX 官方全集地图扩张”：每次 blocked 都会让 support matrix 多一条依据明确的路线，要么新增算子，要么细化已有算子的子形态。

## 下一阶段计划

### M0: ONNX schema 与 support matrix 基线化

目标：

- 穷举当前绑定 ONNX 版本的官方 opset/schema，作为 converter 和 codegen 的共同基线。
- 建立 NanoC-NN support matrix，明确每个 ONNX op 的支持子形态、lowering 策略和拒绝原因。
- 把 TDD case 与 support matrix 行绑定，让能力声明来自表和测试的交叉验证。
- 绑定 `onnx` Python package 版本，避免未来 ONNX 版本升级导致能力边界漂移。

建议工作：

1. 读取模型 `opset_import`，在 converter 输出中记录每个 node 的 `domain`、`op_type`、`since_version`。
2. 使用 ONNX 官方 schema API 生成版本绑定 catalog，保存到 `tdd/onnx_schema/`。
3. 将 catalog 作为官方全集；support matrix 只能引用 catalog 中存在的 official op，真实模型中出现但 catalog 没有的 op 必须标为 extension。
4. 新增 `tdd/support_matrix/` 或 `docs/codegen/support_matrix.md`，记录当前白名单与 planned 行。
5. 定义 support matrix 与 case registry 的字段对应关系。
6. 调整 case 规格模板，要求写明覆盖的 ONNX op schema 子形态。

测试入口：

- `python tdd/scripts/run_tests.py --validate-only`
- 新增 ONNX catalog、support matrix 与 case registry 的一致性检查。
- 对现有 case 先做只读扫描，不先改变生成行为。

通过标准：

- 当前 ONNX 官方 schema catalog 完整生成，并记录 `onnx` 版本与默认 opset。
- `schema_source="official"` 的 support matrix 行必须能在 catalog 中找到。
- 每个 `ok` case 都能追溯到 support matrix 的 `ok` 行。
- 每个 `blocked/unsupported` case 都能追溯到稳定拒绝行。
- 不再出现“代码里支持了某个分支但能力表和测试没有记录”的状态。

### M0.5: 既有 TDD 资产统一迁移

目标：

- 将已有 TDD case 全量迁移到 ONNX schema/support matrix 体系。
- 保留历史 case ID 和回归资产，但消除“旧 case 是历史特例、新 case 才系统化”的断层。
- 让 `CONV_*`、`GEMM_*`、`TOPO_*`、`NET_*`、`TS_*`、`QLINEAR_NUM_*`、`NEG_*` 都能按同一套元数据规则解释。

迁移原则：

- 旧 case ID 不重命名，避免破坏历史记录、报告路径和回归命令。
- 旧 case 必须补齐 schema metadata，而不是继续作为自由文本说明存在。
- 业务名称只作为描述，不作为前端算子分类依据。
- 旧 case 的 expected status 不因迁移被削弱；如果迁移暴露出假阳性，应先保留失败并进入修复队列。

建议工作：

1. 盘点 `tdd/cases/` 下所有 case，生成现状清单。
2. 为每个 case 标注：
   - `domain`
   - `op_type`
   - `opset_range`
   - `schema_form`
   - `lowering`
   - `backend`
   - `capability_type`
   - `support_matrix_id`
3. 更新 `tdd/cases/README.md` 和 case 模板，要求所有旧/新 case 都具备相同字段。
4. 扩展 `tdd/scripts/cases_registry.py`，让 registry 能表达 schema 子形态和 backend/capability type。
5. 新增或扩展 validate-only 检查：
   - case 文件存在于 registry。
   - registry case 能匹配 support matrix 行。
   - support matrix 行至少被一个 case 覆盖，或显式标记为 planned。
   - `ok` case 必须有 required API 或 reference backend 说明。
6. 更新 `tdd/CAPABILITIES.md` 生成逻辑，使能力展示从“case 列表”升级为“ONNX op -> schema 子形态 -> backend -> case”。
7. 对现有 case 做统一解释：
   - `CONV_001~004` 是 ONNX `Conv` 的 rank/group/pad/zp 子形态。
   - `TOPO_002` 是 ONNX `Conv` rank=3 时序子形态组合链路。
   - `TOPO_003` 是 ONNX `QLinearConv`、`QLinearAdd`、`MaxPool`、`Gemm/MatMul` 等子形态组合链路。
   - `TS_*` 是 reference backend 数值验收能力，不等同于 CMSIS-NN int8 lowering。

测试入口：

- `python tdd/scripts/run_tests.py --validate-only`
- `python tdd/scripts/run_tests.py --mode target --generate`
- `python tdd/scripts/run_numeric_tests.py --generate`
- 必要时再跑 `python tdd/scripts/run_regression.py --generate`

通过标准：

- 所有既有 case 都能通过统一 schema/support matrix 元数据校验。
- 旧能力集没有退化。
- 能力报告不再把旧 case 和新 case 分成两种体系。
- 任何新增 case 若缺少 schema/support matrix 绑定，validate-only 必须失败。

### M1: 固化能力口径

目标：

- 明确 `CMSIS-NN int8 交付能力` 与 `C reference 验证能力` 的区别。
- 避免把 TS 系列 float reference pass 误读为 CMSIS-NN int8 pass。
- 将能力口径并入 support matrix，而不是散落在报告和文档中。

测试入口：

- 保持 `run_tests.py --mode target --generate` 作为结构能力入口。
- 保持 `run_numeric_tests.py --generate` 作为 ONNX-vs-C 数值入口。
- 在 case registry 或 support matrix 中增加 backend/capability type 字段。

预期产物：

- 能力集展示分层设计。
- 新增或更新说明文档，明确 reference backend 的用途。

### M2: 重构 codegen emitter 结构

目标：

- 借鉴 onnx2c 的 per-node emitter，把 `generator.py` 中 CMSIS-NN runtime 生成逻辑逐步拆分。
- 每个新增或迁移 emitter 都必须由现有 TDD case 保护。

建议拆分顺序：

1. `emitters/conv.py`
2. `emitters/gemm.py`
3. `emitters/pool.py`
4. `emitters/add.py`
5. `emitters/concat.py`
6. `emitters/softmax.py`
7. `emitters/reference.py`

测试入口：

- 迁移前后先跑对应最小 case，例如 `CONV_*`、`GEMM_*`、`MAXPOOL_001`。
- 每完成一个 emitter 迁移，跑 `python tdd/scripts/run_tests.py --mode target --generate`。
- 最终跑 `python tdd/scripts/run_regression.py --generate`。

通过标准：

- 行为不变。
- 生成物仍包含真实 CMSIS-NN 调用。
- 既有 numeric case 不退化。

### M3: 引入 pass pipeline

目标：

- 在 codegen 中建立显式 pass 阶段，避免 layout、QDQ、fuse、memory、budget 逻辑散落在 emitter 中。

建议 pass：

- `fold_shape_ops`: 处理 Flatten、Reshape、Shape、Gather、Slice 等生成期 shape op。
- `fold_qdq_boundaries`: 识别 QuantizeLinear / DequantizeLinear 的边界和 alias。
- `fuse_activation`: 将 Relu/Clip 显式折叠到 CMSIS-NN activation min/max。
- `normalize_layout`: 明确 NCHW/NHWC/NCW 的输入、输出和 flatten 顺序。
- `plan_lifetime_memory`: 计算 tensor 生命周期和 buffer 复用。
- `check_platform_budget`: 根据 target profile 判断 `oversize`。

测试入口：

- 从现有 MNIST、signal jump、MobileNetV2 case 中提炼最小用例。
- 优先新增 layout/fuse/memory 类用例，不直接盲修完整网络。

通过标准：

- pass 前后报告可追踪。
- 每个 pass 的行为有最小 case 保护。

### M4: 生命周期内存规划

目标：

- 借鉴 onnx2c tensor union 思路，从“两块最大 activation buffer”升级为按 tensor 生命周期复用 buffer。

测试入口：

- 新增 topology/memory 类用例：
  - 单链路 Conv -> Pool -> FC。
  - 分支 Concat。
  - 残差 Add。
  - 多输出但只消费部分输出。

通过标准：

- 生成报告包含每个 tensor 的 buffer 分配和复用关系。
- SRAM 估算不大于当前保守估算。
- 生成 C 仍通过 smoke compile 和数值回归。

### M5: 逐层 dump 与误差定位

目标：

- 支持 ONNX 节点输出和 C 中间 buffer 的逐层对比。
- 降低完整网络失败时的定位成本。

建议能力：

- ONNX Runtime dump 指定节点输出。
- 生成 C 可选开启 `NANOC_TRACE_TENSORS`。
- numeric runner 能按 node/tensor 名称比对 max abs error、top1、饱和率。

优先服务用例：

- `TOPO_003` MNIST。
- `NET_005` signal jump。
- `NET_004` KWS DS-CNN。
- `NET_001` SqueezeNet blocked 分析。

通过标准：

- 至少一个已通过网络能输出逐层对齐报告。
- 至少一个失败/blocked 网络能定位到首个明显误差或 unsupported tensor。

### M6: 真实网络扩展与 SqueezeNet 收敛

目标：

- 继续以真实网络驱动能力扩展。
- 优先处理当前已经暴露的 SqueezeNet 缺口，而不是无目标新增大量网络。

优先提炼的最小用例：

- QDQ + Concat 多分支。
- QLinearGlobalAveragePool。
- QDQ 后 float Softmax 的处理策略。
- depthwise / pointwise conv 组合。
- 多分支布局保持与 concat axis。

测试入口：

- `NET_001` 保持当前 blocked 预期，先从它反向提炼 core/topology 用例。
- 每个最小用例通过后，再回测 `NET_001` 是否能从 blocked 收敛到 ok。

通过标准：

- 不允许仅报告 `ok`，必须检查生成物存在真实可编译推理路径。
- 若仍不支持，必须稳定返回 `blocked` 或 `unsupported`，不能假阳性。

### M7: 平台 profile 与发布门禁

目标：

- 将 Cortex-M3/M4/M7/M55、SRAM/Flash、DSP/MVE/FPU 支持能力显式化。
- 建立发布前最小门禁。

建议 profile：

- `cortex-m3`: 无 DSP/MVE，严格 SRAM/Flash。
- `cortex-m4`: DSP 可用，可跑基础 CMSIS-NN s8。
- `cortex-m7`: 较高 SRAM/Flash，作为大多数测试默认正向平台。
- `cortex-m55`: MVE 路线，后续优化目标。

发布门禁：

- `pytest`
- `run_tests.py --mode target --generate`
- `run_numeric_tests.py --generate`
- `run_regression.py --generate`
- 至少检查一个生成物的 C99 smoke compile。

通过标准：

- 结构能力、数值能力、平台预算能力分层报告。
- release 文档只声明测试获得的能力。

## 优先级建议

短期最优先：

1. ONNX schema 与 support matrix 基线化。
2. 既有 TDD 资产统一迁移。
3. TDD case 与 support matrix 绑定。
4. 能力口径分层。
5. codegen emitter 拆分。
6. pass pipeline 骨架。

中期最优先：

1. 逐层 dump 工具。
2. tensor lifetime memory planner。
3. SqueezeNet 反向最小用例。
4. KWS DS-CNN CMSIS-NN int8 数值闭环。

长期最优先：

1. 平台 profile。
2. 更多真实 MCU 场景网络。
3. release 级工程包质量。

## 执行原则

- 每个 plan 项都必须从 ONNX schema/support matrix 和 TDD case 开始。
- ONNX 官方 op schema 是前端事实，不以业务名称随意创造平行算子。
- 业务子形态必须落在 ONNX op 的属性、rank、类型、量化模式和 layout 条件上。
- 旧 case 不允许作为例外体系保留，必须迁移到同一棵 schema/support matrix case 树。
- 新增能力必须有最小用例和完整网络用例双层保护。
- 正确拒绝也是 PASS，但不能假阳性 `ok`。
- 参考项目只提供工程启发，不引入直接依赖。
- `tdd/work/comparative_research/` 仅为临时研究区，不作为正式源码或测试资产。
