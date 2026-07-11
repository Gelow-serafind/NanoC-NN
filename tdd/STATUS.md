# TDD 状态总览

> 本文件是 agent 进入 `tdd/` 目录后的第一个入口。阅读此文件即可了解当前能力边界、最近验证结果和下一步工作方向。

## 当前结论

NanoC-NN 当前已经进入“真实完整网络驱动”的 TDD 阶段。

截至 2026-07-11：

- target 结构回归：`33/33 PASS`
- 稳定回归入口：`python tdd/scripts/run_regression.py --generate` 已通过，baseline 结构 `15/15 PASS`
- 数值回归：`11/11 PASS`，`NET_001` SqueezeNet 已加入 ONNX-vs-C 并行推理验收，`top1=4/4`，饱和率 `0.00`，最大绝对误差 `0.2`
- 真实网络导入测试扩展到 `NET_001~NET_006`
- 新增/晋升结构 `ok` 网络：`NET_001` SqueezeNet 1.0 int8、`NET_002` MobileNetV2 int8、`NET_004` KWS DS-CNN-style、`NET_005` signal jump int8
- 新增正确拒绝网络：`NET_003` SSD-MobileNet int8、`NET_006` EfficientNet-Lite4 int8，当前为 `unsupported`
- 平台预算拦截已从 `blocked` 拆出为 `oversize`
- ONNX 官方 `Concat` 已按 schema-driven TDD 进入能力集，当前确认子形态为 rank=4 NCHW、`axis=1` channel concat、同量化 QDQ/int8，lowering 到 `arm_concatenation_s8_z`
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
| 测试用例总数 | 33 |
| target 通过 | 33 |
| target 失败 | 0 |
| 稳定结构回归 | 15/15 PASS |
| 数值回归 | 11/11 PASS |
| 当前能力集文件 | `tdd/CAPABILITIES.md` |
| 当前可视化图谱 | `tdd/reports/onnx_support_map.html` |
| 最新迭代记录 | `tdd/iterations/019_squeezenet_numeric_pass.md` |

## 已确认主线能力

| 能力 | 代表用例 | 当前结论 |
|------|----------|----------|
| Conv / QLinearConv CMSIS-NN 生成 | `CONV_001~004`, `QLINEAR_NUM_001~003` | PASS |
| Gemm / MatMul / FC 生成 | `GEMM_001~004` | PASS |
| MaxPool int8 生成 | `MAXPOOL_001`, `MAXPOOL_002` | PASS |
| QLinearGlobalAveragePool int8 生成 | `AVGPOOL_001` | PASS |
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

# 全量 target：更新 CAPABILITIES.md + latest.json
python tdd/scripts/run_tests.py --mode target --generate

# 生成 ONNX 支持思维导图
python tdd/scripts/generate_support_map.py

# MNIST 数值验收
python tdd/scripts/run_numeric_tests.py --case TOPO_003 --dataset mnist_synthetic_smoke --generate

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
7. 为 Conv1d NCW flatten 和 Conv/Relu Q/DQ 折叠补独立最小数值用例，降低完整网络失败时的定位成本。
