# 迭代 010: MNIST 后续回归与能力扩展计划

## 基本信息

- **日期**: 2026-07-04
- **前置迭代**: 009
- **触发来源**: 人类需求 / MNIST int8 已打通后的防退化与能力扩展

## 目标

MNIST int8 的 C 代码生成已经从结构通过推进到 ONNX-vs-C 数值一致。下一阶段不应继续围绕同一个模型盲目堆需求，而应把本轮暴露过的真实缺陷沉淀为稳定测试能力，并逐步扩展到低算力嵌入式平台常见的小型网络。

本轮计划拆成四条主线：

1. 将 `TOPO_003` 数值测试纳入常规回归，防止布局和量化修复退化。
2. 增加逐层 dump 工具，支持 ONNX 节点输出与 C 中间 buffer 对比。
3. 为 `QLinearAdd`、`Flatten/Reshape`、per-channel FC 分别补独立最小数值用例。
4. 增加真实小型模型族，覆盖 keyword spotting、tiny anomaly detection、简单 IMU 分类等 MCU 典型场景。

## 新增/修改的测试用例

### 1. 数值回归门禁

修改目标：

- `tdd/scripts/run_tests.py` 或新增统一入口，使常规回归可以包含 numeric 层。
- `TOPO_003` 至少默认跑 `mnist_synthetic_smoke`。
- 手写数据集 `mnist_hand_drawn` 保持可手动指定运行，不直接作为所有环境的必跑项，避免人工样本变化导致门禁波动。

预期命令形态：

```bash
python tdd/scripts/run_tests.py --mode target --generate
python tdd/scripts/run_numeric_tests.py --case TOPO_003 --dataset mnist_synthetic_smoke --generate
```

后续可考虑增加组合入口：

```bash
python tdd/scripts/run_regression.py --generate
```

验收标准：

- 结构回归继续保持全量 PASS。
- `TOPO_003 + mnist_synthetic_smoke` top1 一致率为 100%。
- C 输出饱和比例不超过当前阈值。
- 失败时报告中明确区分 structural fail、compile fail、run fail、numeric mismatch。

### 2. 逐层 dump 工具

新增目标：

- 在数值测试失败时，能够选择性输出 ONNX 中间节点结果与 C 中间 buffer。
- 工具优先服务 TDD 定位，不要求进入嵌入式最终生成物 API。

建议文件：

```text
tdd/scripts/dump_numeric_trace.py
tdd/work/numeric/<case_id>/trace/
```

设计约束：

- ONNX dump 通过 ONNX Runtime session 或临时改图暴露中间 tensor。
- C dump 通过测试 runner 编译时宏启用，例如 `NANOC_TRACE_TENSORS`。
- 生成 C 的正式推理路径默认不输出、不依赖文件系统、不引入 malloc。
- trace 文件只写入 `tdd/work/`，不作为永久资产；若某次 trace 有长期价值，再人工提升到 `tdd/results/history/` 或 `tdd/iterations/`。

第一阶段覆盖节点：

- Conv1 output
- Pool1 output
- Conv2 output
- Pool2 output
- Flatten/FC input
- QLinearMatMul output
- QLinearAdd output

验收标准：

- 对 `TOPO_003` 可一键生成 trace。
- trace index、shape、scale、zero point、top-k 或摘要统计可读。
- 当 ONNX 与 C top1 不一致时，能定位第一个明显偏离的层。

### 3. 独立最小数值用例

MNIST 证明了完整网络闭环，但完整网络失败时定位成本高。需要将这次真实问题反向拆成三个最小用例。

#### QLINEAR_ADD_NUM_004: QLinearAdd scale 差异与 left_shift 回归

覆盖问题：

- 防止 `left_shift=20` 一类固定放大再次导致饱和。
- 覆盖一边是 activation、一边是 bias/constant 的场景。

验收标准：

- ONNX-vs-C top1 或逐元素输出一致。
- 饱和比例低于阈值。
- 量化参数报告中明确记录 `left_shift`、input multiplier/shift、output offset。

#### FLATTEN_NUM_001: NHWC runtime 到 NCHW flatten 语义

覆盖问题：

- 防止 CMSIS-NN Conv/Pool 输出 buffer 直接 alias 给 FC。
- 验证 `4D -> 2D` flatten 的数据顺序与 ONNX 一致。

建议模型：

```text
QLinearConv/Conv -> MaxPool 可选 -> Reshape/Flatten -> QLinearMatMul
```

验收标准：

- 构造输入使不同 channel/height/width 位置有可区分值。
- C 侧 FC 输入顺序与 ONNX flatten 顺序一致。
- 生成物中出现显式 flatten transform buffer 或等效逻辑。

#### FC_PER_CHANNEL_NUM_001: QLinearMatMul per-channel FC

覆盖问题：

- 防止 per-channel weight scale 被压成 common scale。
- 防止退回 `arm_fully_connected_s8`。

验收标准：

- 生成物调用 `arm_fully_connected_per_channel_s8`。
- multiplier/shift 以数组形式生成。
- ONNX-vs-C 输出一致或在明确阈值内。

### 4. 真实小型模型族

扩展模型必须符合项目定位：低算力 Arm Cortex-M 平台、int8 量化、可作为用户真实输入的 ONNX。优先小模型、固定输入尺寸、无动态 shape。

候选模型族：

| 模型族 | 场景 | 优先原因 | 首轮验收 |
|--------|------|----------|----------|
| keyword spotting | 低功耗语音唤醒 | MCU 常见场景，通常是小型 DS-CNN/TCN | 先跑 converter/codegen，记录不支持算子 |
| tiny anomaly detection | 设备状态/传感器异常 | 输入尺寸小，适合裸机周期推理 | 优先找全连接或小 1D CNN |
| simple IMU classification | 姿态/动作识别 | 低算力传感器分类典型应用 | 可用小 MLP/1D CNN |
| tiny vision classifier | 低分辨率图像分类 | 与 MNIST 相近但类别/结构不同 | 作为 MNIST 外第二个视觉闭环 |

模型来源优先级：

1. 已有公开 int8 ONNX。
2. 可由公开 PyTorch/TensorFlow 小模型稳定导出 ONNX 并量化。
3. 项目内训练一个极小模型，但必须说明数据来源、训练脚本和量化方式。

验收阶段：

- 阶段 A：下载或生成 ONNX fixture，记录来源、license、输入输出语义。
- 阶段 B：运行 converter/codegen，得到支持/不支持算子列表。
- 阶段 C：只对已经进入能力白名单的模型做数值测试。
- 阶段 D：将真实失败反向拆成最小用例，再进入源码修复。

## 执行结果

本轮已开始执行：

- 新增 `NET_001`，导入真实 SqueezeNet 1.0 int8 ONNX fixture。
- `NET_001` 首次执行结果：PASS，`expected=blocked`，`actual=blocked`。
- 新增 `run_regression.py` 稳定回归入口，组合 baseline 结构验收与已登记 numeric 验收。
- `run_regression.py --generate` 执行结果：PASS。

已执行命令：

```bash
python tdd/scripts/run_tests.py --validate-only
python tdd/scripts/run_tests.py --case NET_001 --generate
python tdd/scripts/run_regression.py --generate
python tdd/scripts/run_tests.py --mode target --generate
python -m nanoc_nn.cli onnx-to-cmsis \
  --model tdd/fixtures/onnx/mnist-12-int8.onnx \
  --target cortex-m3 \
  --sram-budget 1K \
  --flash-budget 1M \
  --out-root tdd/work/platform_probe/TOPO_003_oversize \
  --no-compile
```

当前结果：

```text
validate: 18 cases PASS
NET_001: expected=blocked actual=blocked PASS
baseline structural: 3/3 PASS
numeric: 1/1 PASS (TOPO_003 top1=10/10, sat=0.00)
regression: PASS
target: 18/18 PASS
platform budget probe: status=oversize, mappings_blocked_or_unsupported=0, quantization_issues=0
```

## 代码修改

- `tdd/scripts/cases_registry.py`: 注册 `NET_001`。
- `tdd/scripts/generate_models.py`: 增加从真实 fixture 复制 `NET_001` 的生成函数。
- `tdd/scripts/validate_cases.py`: 允许 `cases/` 下存在非 case 的专题文档，避免 MNIST 复盘文档被误识别为规格文件。
- `tdd/scripts/run_regression.py`: 新增稳定回归入口。
- `src/nanoc_nn/codegen/generator.py`: 将预算超限从 `blocked` 中拆出为 `oversize`。
- `tdd/scripts/run_tests.py`: 常规结构测试不再默认传入 SRAM/Flash 预算，避免把平台适配问题混入 ONNX 生成能力测试。
- `tdd/cases/networks/NET_001.md`: 新增真实 SqueezeNet int8 导入边界规格。
- `tdd/fixtures/onnx/MANIFEST.md`: 记录真实 ONNX fixture 来源、license、SHA256 和首次导入结论。

## 最终结果

第一阶段已完成：

- `TOPO_003` 数值测试已进入新的稳定回归入口。
- `NET_001` 作为第一个 MNIST 之外的真实 ONNX 导入测试进入 TDD。
- SqueezeNet 当前没有被误判为支持；它被正确 blocked，并暴露后续真实能力缺口。
- 全量 target 已扩展到 18 个用例并全部通过。
- `oversize` 已从 `blocked` 中拆出：当代码生成语义和量化检查通过，但显式传入的目标 SRAM/Flash 预算不足时，codegen 返回 `oversize`。

本轮尚未修复 SqueezeNet 的 blocked 原因。后续应优先从报告中提炼最小用例，而不是直接在完整 SqueezeNet 上盲修。

## 发现与后续

本轮后续开发应遵循以下顺序：

1. 先把 `TOPO_003` numeric 纳入可重复回归入口。
2. 再做逐层 dump 工具，提升定位能力。
3. 再补三个最小数值用例，固化 MNIST 暴露过的真实缺陷。
4. 最后引入真实小型模型族，按“不支持原因 -> 最小复现 -> 修复 -> 数值验收”的节奏扩展能力集。

这条路线的核心是：不再凭空创造需求，而是把真实模型暴露的问题拆小、固化、回归，然后再用新的真实模型继续扩大边界。

`NET_001` 给出的下一批真实缺口：

- float-domain `Concat` 前后穿插 Q/DQ 时的 layout 与量化处理需要更小用例验证。
- `QLinearGlobalAveragePool` 虽然 op mapping 有 API 映射，但量化字段仍存在缺口，需要独立最小用例。
- Softmax 出现在 float output 侧时，当前量化字段提取仍不足，需要确认是应折叠、应拒绝还是应生成真实 int8 softmax。
- 真实视觉模型的 SRAM 估算可能远超某些 MCU 预算，但这不应混入 ONNX 导入能力判断。后续要持续区分“算子/量化/生成语义不完整”的 `blocked` 与“目标内存预算不满足”的 `oversize`。常规 ONNX 导入测试以代码生成完整性和数值正确性为主，平台预算只在显式传入目标约束时拦截。
