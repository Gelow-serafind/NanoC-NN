# TDD 状态总览

> 本文件是 agent 进入 `tdd/` 目录后的第一个入口。阅读此文件即可了解当前能力边界、最近验证结果和下一步工作方向。

## 当前结论

NanoC-NN 当前已经进入“真实完整网络驱动”的 TDD 阶段。

截至 2026-08-11：

- target 结构回归：`82/82 PASS`
- 稳定回归入口：`PYTHONPATH=src python tdd/scripts/run_regression.py --generate` 已通过，baseline 结构 `60/60 PASS`
- 数值回归：`60/60 PASS`，本轮新增 `GreaterOrEqual`、`LessOrEqual`、`And`、`Or`、`Not`、`Xor` 六个官方 ONNX 算子的最小数值验收；新增六项均为 `top1=2/2`、饱和率 `0.00`、最大绝对误差 `0.0`
- 终端实机验收层已建立：`tdd/terminal/` 支持 ONNX-vs-HostC-vs-ARMC 三路对比，当前首个接入目标为 `TOPO_003` MNIST int8 on STM32F103
- 真实网络导入测试扩展到 `NET_001~NET_006`
- 新增/晋升结构 `ok` 网络：`NET_001` SqueezeNet 1.0 int8、`NET_002` MobileNetV2 int8、`NET_004` KWS DS-CNN-style、`NET_005` signal jump int8
- 新增正确拒绝网络：`NET_003` SSD-MobileNet int8、`NET_006` EfficientNet-Lite4 int8，当前为 `unsupported`
- 平台预算拦截已从 `blocked` 拆出为 `oversize`
- ONNX 官方 `Concat` 已按 schema-driven TDD 进入能力集，当前确认子形态为 rank=4 NCHW、`axis=1` channel concat、同量化 QDQ/int8，lowering 到 `arm_concatenation_s8_z`
- ONNX 官方 `GlobalAveragePool` 已按 schema-driven TDD 进入能力集，当前确认子形态为 rank=4 NCHW、QDQ/int8、全局 H/W 池化，lowering 到 `arm_avgpool_s8`
- ONNX 官方 `Flatten` 已按 schema-driven TDD 进入能力集，当前确认子形态为 QDQ/int8、`axis=1`、保持 ONNX row-major flatten 顺序；无 CMSIS 算术 kernel，codegen 生成 layout-only copy 路径。
- ONNX 官方 `MatMul` 已按 schema-driven TDD 进入能力集，当前确认子形态为 `A[1,I] x B[I,O] -> Y[1,O]`、QDQ/int8、静态常量权重，lowering 到 `arm_fully_connected_s8`。
- ONNX 官方 `Abs` 已按 schema-driven TDD 进入能力集，当前确认子形态为静态 QDQ/int8 tensor、输入输出同 scale/zero_point；CMSIS-NN 无专用 s8 kernel，codegen 生成 C99 逐元素 abs 路径并将 `-128` clamp 到 `127`。
- ONNX 官方 `AveragePool` 已按 schema-driven TDD 进入能力集，当前确认子形态为 rank=4 NCHW、QDQ/int8、2x2 stride=2 无 padding，lowering 到 `arm_avgpool_s8`。
- ONNX 官方 `Reshape` 和 `Squeeze` 已按 schema-driven TDD 进入能力集，当前确认静态 shape、元素数量不变、线性存储顺序不变的 QDQ/int8 shape-only copy/fold 路径。
- ONNX 官方 `Add` 和 `Mul` 已按 schema-driven TDD 进入能力集，当前确认同形状 QDQ/int8、第二输入可为量化常量，分别 lowering 到 `arm_elementwise_add_s8` 和 `arm_elementwise_mul_s8`。
- ONNX 官方 `Sub` 和 `Div` 已按 schema-driven TDD 进入能力集，当前确认同形状 QDQ/int8、第二输入可为量化常量；CMSIS-NN 无专用 s8 kernel，codegen 生成 C99 逐元素反量化、算术、再量化路径。
- ONNX 官方 `Sigmoid` 已按 schema-driven TDD 进入能力集，当前确认静态 QDQ/int8 tensor，codegen 生成 C99 `expf` 正确性基线路径，后续可替换为 LUT 或定点近似。
- ONNX 官方 `Pad`、`Slice`、`Gather` 已按 schema-driven TDD 进入能力集，当前确认静态参数、同量化 QDQ/int8 copy/indexing 路径；`Slice/Gather` 的 shape-helper 形态仍维持 generation-time fold。
- ONNX 官方 `Transpose` 已按 schema-driven TDD 进入能力集，当前确认 rank=4 QDQ/int8 显式 `perm=[0,2,3,1]`，lowering 到 `arm_transpose_s8`。
- ONNX 官方 `Tanh`、`LeakyRelu`、`Clip`、`Neg`、`Sqrt`、`Reciprocal` 已按 schema-driven TDD 进入能力集，当前确认静态 QDQ/int8 tensor，codegen 生成 C99 逐元素 correctness baseline；`Clip` 已拆出 standalone data-path，避免误用旧 activation alias。
- ONNX 官方 `ReduceMean` 已按 schema-driven TDD 进入能力集，当前确认 rank=2、`axes=[1]`、`keepdims=1` 的 QDQ/int8 行均值路径。
- ONNX 官方 `Unsqueeze` 数据路径已按 schema-driven TDD 进入能力集，当前确认静态 QDQ/int8、常量 axes、元素数量不变的 copy 路径；shape-helper Unsqueeze 仍保持 generation-time fold。
- ONNX 官方 `Exp`、`Log`、`Floor`、`Ceil`、`Round`、`Sign` 已按 schema-driven TDD 进入能力集，当前确认静态 QDQ/int8 tensor，codegen 生成 C99 逐元素 correctness baseline。
- ONNX 官方 `Min`、`Max`、`Pow` 已按 schema-driven TDD 进入能力集，当前确认同形状 QDQ/int8、第二输入可为量化常量；CMSIS-NN 无专用 s8 kernel，codegen 生成 C99 逐元素反量化、算术、再量化路径。
- ONNX 官方 `ReduceSum`、`ReduceMax`、`ReduceMin` 已按 schema-driven TDD 进入能力集，当前确认 rank=2、`axes=[1]`、`keepdims=1` 的 QDQ/int8 行归约路径。
- ONNX 官方 `Where` 已按 schema-driven TDD 进入能力集，当前确认静态 bool mask、同形状 QDQ/int8 then/else 数据选择，codegen 生成 C99 逐元素选择路径。
- ONNX 官方 `Equal`、`Greater`、`Less` 已按 schema-driven TDD 进入能力集，当前确认同形状 QDQ/int8 比较，bool 输出作为中间 condition 被 `Where` 消费。
- ONNX 官方 `GreaterOrEqual`、`LessOrEqual` 已按 schema-driven TDD 进入能力集，当前确认同形状 QDQ/int8 比较（含边界相等），bool 输出被 `Where` 消费。
- ONNX 官方 `And`、`Or`、`Not`、`Xor` 已按 schema-driven TDD 进入能力集，当前确认 bool 输入由 compare 算子产出、bool 输出被 `Where` 消费，codegen 生成 C99 逐元素逻辑路径。
- ONNX 官方 `ReduceProd`、`ReduceL1`、`ReduceL2` 已按 schema-driven TDD 进入能力集，当前确认 rank=2、`axes=[1]`、`keepdims=1` 的 QDQ/int8 行归约路径。
- SqueezeNet 暴露出的 Microsoft 扩展 `QLinearGlobalAveragePool` 已提炼为独立最小用例 `AVGPOOL_001`，当前确认静态 rank=4 NCHW、全局 H/W 池化，lowering 到 `arm_avgpool_s8`
- `NET_001` SqueezeNet 1.0 int8 已完成结构与数值闭环：完整网络可生成真实 CMSIS-NN C 工程，通过 C99 smoke compile/run，并在统一 smoke 数据集上通过 ONNX-vs-C 并行推理验收。

## 状态语义

| 状态 | 含义 |
|------|------|
| `ok` | converter/codegen 语义通过，生成物包含当前支持范围内的真实 CMSIS-NN 调用 |
| `blocked` | 算子、量化字段、布局、renderer 或生成语义尚不完整 |
| `unsupported` | converter 或 codegen 明确不支持该 ONNX 形态 |
| `oversize` | 代码生成语义已通过，但显式传入的目标 SRAM/Flash 预算不满足 |

常规 ONNX 能力测试以代码生成完整性和数值正确性为主，不默认使用某个 MCU 的 SRAM/Flash 预算作为能力门槛。目标平台预算只在显式传入 `--sram-budget` / `--flash-budget` 时拦截，并返回 `oversize`。

## 能力集摘要

| 指标 | 当前值 |
|------|--------|
| 测试用例总数 | 82 |
| target 通过 | 82 |
| target 失败 | 0 |
| 稳定结构回归 | 60/60 PASS |
| 数值回归 | 60/60 PASS |
| 终端实机验收 | 可选门禁，接入 STM32 时执行 |
| 当前能力集文件 | `tdd/CAPABILITIES.md` |
| 当前可视化图谱 | `tdd/reports/onnx_support_map.html` |
| 最新迭代记录 | `tdd/iterations/028_graph_bool_logic_compare_ext.md` |

## 已确认主线能力

| 能力 | 代表用例 | 当前结论 |
|------|----------|----------|
| Conv / QLinearConv CMSIS-NN 生成 | `CONV_001~004`, `QLINEAR_NUM_001~003` | PASS |
| Gemm / MatMul / FC 生成 | `GEMM_001~004`, `MATMUL_001` | PASS |
| Abs int8 逐元素绝对值 | `ABS_001` | 结构 PASS + 数值 PASS |
| MaxPool int8 生成 | `MAXPOOL_001`, `MAXPOOL_002` | PASS |
| AveragePool int8 生成 | `AVGPOOL_003` | 结构 PASS + 数值 PASS |
| GlobalAveragePool int8 生成 | `AVGPOOL_002` | 结构 PASS + 数值 PASS |
| QLinearGlobalAveragePool int8 生成 | `AVGPOOL_001` | PASS |
| Flatten / Reshape / Squeeze int8 布局折叠 / copy | `FLATTEN_001`, `RESHAPE_001`, `SQUEEZE_001` | 结构 PASS + 数值 PASS |
| Add / Mul / Sub / Div int8 逐元素算术 | `ADD_001`, `MUL_001`, `SUB_001`, `DIV_001` | 结构 PASS + 数值 PASS |
| Sigmoid / Tanh / LeakyRelu int8 逐元素激活 | `SIGMOID_001`, `TANH_001`, `LEAKYRELU_001` | 结构 PASS + 数值 PASS |
| Clip / Neg / Sqrt / Reciprocal int8 逐元素生成路径 | `CLIP_001`, `NEGOP_001`, `SQRT_001`, `RECIPROCAL_001` | 结构 PASS + 数值 PASS |
| Exp / Log / Floor / Ceil / Round / Sign int8 逐元素生成路径 | `EXP_001`, `LOG_001`, `FLOOR_001`, `CEIL_001`, `ROUND_001`, `SIGN_001` | 结构 PASS + 数值 PASS |
| Min / Max / Pow int8 逐元素算术 | `MIN_001`, `MAX_001`, `POW_001` | 结构 PASS + 数值 PASS |
| Where int8 条件选择 | `WHERE_001` | 结构 PASS + 数值 PASS |
| Equal / Greater / Less bool 中间比较 | `EQUAL_001`, `GREATER_001`, `LESS_001` | 结构 PASS + 数值 PASS |
| GreaterOrEqual / LessOrEqual bool 中间比较 | `GREATEROREQUAL_001`, `LESSOREQUAL_001` | 结构 PASS + 数值 PASS |
| And / Or / Not / Xor bool 逻辑中间张量 | `AND_001`, `OR_001`, `NOT_001`, `XOR_001` | 结构 PASS + 数值 PASS |
| ReduceMean / ReduceSum / ReduceMax / ReduceMin / ReduceProd / ReduceL1 / ReduceL2 int8 行归约 | `REDUCEMEAN_001`, `REDUCESUM_001`, `REDUCEMAX_001`, `REDUCEMIN_001`, `REDUCEPROD_001`, `REDUCEL1_001`, `REDUCEL2_001` | 结构 PASS + 数值 PASS |
| Pad / Slice / Gather int8 静态索引 copy | `PAD_001`, `SLICE_001`, `GATHER_001` | 结构 PASS + 数值 PASS |
| Unsqueeze int8 静态升维 copy | `UNSQUEEZE_001` | 结构 PASS + 数值 PASS |
| Transpose int8 显式置换 | `TRANSPOSE_001` | 结构 PASS + 数值 PASS |
| Softmax int8 生成 | `SOFTMAX_001`, `SOFTMAX_002` | PASS，`SOFTMAX_002` 数值 PASS |
| Concat int8 channel 拼接 | `CONCAT_001`, `CONCAT_002` | 结构 PASS + 数值 PASS |
| 组合拓扑生成 | `TOPO_001`, `TOPO_002` | PASS |
| 真实 MNIST QLinear int8 完整网络 | `TOPO_003` | 结构 PASS + 数值 PASS |
| float32 无 Q/DQ 模型拒绝 | `NEG_001` | PASS |
| 真实 SqueezeNet int8 完整网络 | `NET_001` | 结构 PASS + 数值 PASS |
| 真实 MobileNetV2 int8 结构生成 | `NET_002` | 结构 PASS |
| 真实 SSD-MobileNet int8 检测边界 | `NET_003` | 正确 unsupported |
| KWS DS-CNN-style int8 结构生成 | `NET_004` | 结构 PASS |
| Tiny signal jump int8 完整数值闭环 | `NET_005` | 结构 PASS + 数值 PASS |
| EfficientNet-Lite4 int8 边界 | `NET_006` | 正确 unsupported |

## 当前真实网络

### TOPO_003: MNIST int8

`TOPO_003` 是当前第一个完整打通的真实网络：

- ONNX fixture: `tdd/fixtures/onnx/mnist-12-int8.onnx`
- 数据集：
  - `tdd/fixtures/datasets/mnist_synthetic_smoke/`
  - `tdd/fixtures/datasets/mnist_hand_drawn/`
- 数值验收：ONNX Runtime 与生成 C 并行推理一致
- 终端实机验收：`tdd/terminal/cases/TOPO_003_mnist_int8.json` 绑定 STM32F103 runner，目标是 ONNX / Host C / ARM C 三路一致
- 关键修复：`auto_pad`、NHWC 到 NCHW flatten、per-channel FC、`QLinearAdd left_shift=0`
- 复盘文档：`tdd/cases/topology/TOPO_003_mnist/CODEGEN_CASE_STUDY.md`

### NET_001: SqueezeNet 1.0 int8

`NET_001` 是 MNIST 之后第一个真实外部 ONNX 导入测试：

- ONNX fixture: `tdd/fixtures/onnx/squeezenet1.0-12-int8.onnx`
- 来源记录：`tdd/fixtures/onnx/MANIFEST.md`
- 当前预期：`ok`
- 当前意义：确认真实 SqueezeNet fire module 网络能被导入、解析，并生成包含真实 CMSIS-NN 调用的 C 工程。生成物已通过 C99 smoke compile/run，并完成 ONNX-vs-C 并行推理数值验收。
- 数据集：`tdd/fixtures/datasets/net_001_squeezenet_smoke/`
- 数值验收：`top1=4/4`，饱和率 `0.00`，最大绝对误差 `0.2`
- 判定说明：`midgray` 是低置信 near-tie 样本，ONNX top1 与 C top1 在 ONNX 概率上的 margin 为 `0.0174`，低于 `top1_tie_margin=0.02`，因此计入可接受一致；其余 3 个样本 exact top1 一致。

`NET_001` 本轮收敛出的关键能力：

- `DequantizeLinear -> Concat -> QuantizeLinear` 的不同量化分支拼接，由 `CONCAT_002` 独立保护并完成数值验收。
- `Concat -> MaxPool -> QuantizeLinear` 边界处缺失输入量化信息时的反向传播，由 `MAXPOOL_002` 独立保护。
- Microsoft 扩展 `QLinearGlobalAveragePool` 由 `AVGPOOL_001` 独立保护。
- float output 侧 `Softmax` 折叠为 C 侧 int8 softmax 输出，由 `SOFTMAX_002` 独立保护并完成数值验收。
- CMSIS-NN `arm_concatenation_s8_z` 的 block-copy 语义不适合内部 NHWC channel concat，codegen 已对 channel concat 生成显式 NHWC channel-copy 循环，同时保留 API 映射报告。
- 外部 NCHW 输出与内部 NHWC buffer 的边界转置已修正，避免 `return` 前顺序拷贝导致数值验收误判。

仍需后续补齐：

- 明确具体 Cortex-M3/M4/M7 SRAM/Flash 预算下的 `oversize` 门禁。
- 增加逐层 dump 工具，用于完整网络误差定位。
- 追加带真实标签的 ImageNet 小样本，区分“ONNX-vs-C 一致性”与“模型真实分类准确率”。

### NET_002~NET_006: 新一轮真实嵌入式网络

本轮新增 5 个真实或典型嵌入式网络 fixture：

| 用例 | 网络 | 当前结论 | 覆盖重点 |
|------|------|----------|----------|
| `NET_002` | MobileNetV2 int8/QLinear | 结构 PASS | depthwise conv、QLinearAdd、QLinearGlobalAveragePool、QLinearMatMul |
| `NET_003` | SSD-MobileNet int8 | 正确 unsupported | 多输出检测、动态 shape、Loop、检测后处理 |
| `NET_004` | KWS DS-CNN-style int8 | 结构 PASS | 音频特征、depthwise separable conv、pool、FC |
| `NET_005` | tiny signal jump int8 | 结构 PASS | Conv1d、Flatten、FC、时序分类 |
| `NET_006` | EfficientNet-Lite4 int8 | 正确 unsupported | NHWC、QLinearAveragePool、Squeeze、QLinearMatMul 边界 |

注意：`NET_002/004` 当前是结构验收 PASS，即生成物包含真实 CMSIS-NN 调用且 C99 smoke compile/run 通过；它们尚未完成 ONNX-vs-C 数值验收。`NET_005` 已完成 ONNX-vs-C 数值验收。

### NET_005: Tiny signal jump int8

`NET_005` 是内部构造的典型嵌入式时序分类网络，不是外部下载模型：

- ONNX fixture: `tdd/fixtures/onnx/signal_jump.int8.onnx`
- 数据集：`tdd/fixtures/datasets/net_005_signal_jump_smoke/`
- 网络结构：`Conv1d -> Relu -> Conv1d -> Relu -> Flatten -> Gemm`
- 数值验收：ONNX Runtime 与生成 C 并行推理一致
- 当前结果：`top1=3/3`，C label accuracy `1.00`，饱和率 `0.00`，最大绝对误差 `0.0`

本轮关键修复：

- 支持 Conv1d runtime NHWC-like buffer 到 ONNX NCW flatten 顺序的显式转换。
- 支持 Conv/Depthwise 输出经 Q/DQ 进入 Relu 时，将 Relu 折叠为 CMSIS activation min clamp。

## 推荐命令

```bash
# 校验 TDD 用例规格
python tdd/scripts/run_tests.py --validate-only

# 稳定回归：baseline 结构验收 + 已登记 numeric 验收
python tdd/scripts/run_regression.py --generate

# 稳定回归 + 可选 ARM 终端实机验收；无板卡时自动跳过
python tdd/scripts/run_regression.py --generate --terminal auto

# 全量 target：更新 CAPABILITIES.md + latest.json
python tdd/scripts/run_tests.py --mode target --generate

# 生成 ONNX 支持思维导图
python tdd/scripts/generate_support_map.py

# MNIST 数值验收
python tdd/scripts/run_numeric_tests.py --case TOPO_003 --dataset mnist_synthetic_smoke --generate

# MNIST 终端实机验收：ONNX vs Host C vs ARM C
python tdd/terminal/scripts/run_terminal_tests.py --case TOPO_003 --generate --flash --require-board

# 手写 MNIST 数据集采集 UI
python tdd/tools/mnist_capture/server.py
```

## 下一步方向

优先按真实模型暴露的问题继续 TDD 收敛：

1. 为 `NET_001` 追加带真实标签的 ImageNet 小样本，观察模型分类准确率，而不仅是 ONNX-vs-C 一致性。
2. 为 `AVGPOOL_001` 增加独立 ONNX-vs-C 数值验收，或确认 ONNX Runtime 对 Microsoft 扩展 op 的可执行策略。
3. 增加逐层 dump 工具，用于 ONNX 节点输出与 C 中间 buffer 对比。
4. 继续导入 keyword spotting、tiny anomaly detection、简单 IMU 分类等 MCU 常见小模型。
5. 将 `NET_004` KWS-style 升级为数值验收候选。
6. 为 `NET_002` 和 `NET_001` 增加显式 SRAM/Flash 预算 probe，验证 Cortex-M3/M4/M7 平台门禁返回 `oversize` 而不是混入结构能力判断。
7. 继续沿 ONNX 官方图谱推进下一批 MCU 常见算子形态。`GreaterOrEqual/LessOrEqual` 与 `And/Or/Not/Xor` 已在 028 完成；后续优先 `ArgMax/ArgMin` 分类 index 输出（需扩展非 int8 外部 ABI 与 numeric runner），再补 bool initializer 直接作为逻辑算子输入、bool 逻辑算子多层级联。
