# MNIST int8 代码生成案例复盘

## 背景

`TOPO_003` 是 NanoC-NN TDD 中第一条真实完整网络数值验收用例。它使用 `mnist-12-int8.onnx` 作为输入，目标是验证完整链路：

```text
int8/QLinear ONNX -> converter 标准中间表示 -> CMSIS-NN C codegen -> 可运行推理 C 代码
```

该模型不是单算子合成图，而是一个真实 MNIST 分类网络。ONNX 边界输入输出为 `float32`，内部主链路使用 QLinear 量化算子，覆盖 `QLinearConv`、`MaxPool`、`Reshape`、`QLinearMatMul`、`QLinearAdd` 和 `DequantizeLinear`。

MNIST 被引入时，项目原有结构验收只能证明 converter/codegen 能跑通、生成物能编译和启动，不能证明生成 C 的推理结果与原始 ONNX 一致。该模型推动 TDD 验收从“结构通过”升级为“ONNX Runtime 与生成 C 并行推理一致”。

## 需求与验收口径

本用例的最终验收不是 `pipeline_status=ok`，而是以下条件同时满足：

- converter 能解析真实 `mnist-12-int8.onnx` 中的 QLinear 图结构。
- codegen 生成的 `model.c` 包含真实 CMSIS-NN 调用，而不是占位路径。
- 生成 C 可在 host 端链接 `third_party/CMSIS-NN` 编译并运行。
- 同一批输入经过 ONNX Runtime 与生成 C 后，top1 结果一致。
- 输出不再出现全量饱和，例如大量落在 `-128/127`。
- 对带标签数据集，ONNX label accuracy 与 C label accuracy 的差异应可解释且不得明显偏离。

当前已固化的数值入口：

```bash
python tdd/scripts/run_numeric_tests.py --case TOPO_003 --dataset mnist_synthetic_smoke --generate
python tdd/scripts/run_numeric_tests.py --case TOPO_003 --dataset mnist_hand_drawn --generate
```

## 迭代过程摘要

### 1. 引入真实模型后暴露“假成功”

MNIST 初次进入 TDD 时，报告层可能给出 `ok`，但生成物没有包含该网络必需的真实 CMSIS-NN API。此阶段固化了一个关键规则：判断 codegen 成功必须查看生成物本身，包括 `model.c` 中的 API、C99 编译、运行 smoke，而不能只看 converter/codegen 报告状态。

当时需要出现的核心 API 包括：

- `arm_convolve_wrapper_s8`
- `arm_max_pool_s8`
- `arm_fully_connected_s8`，后续升级为 `arm_fully_connected_per_channel_s8`
- `arm_elementwise_add_s8`

### 2. QLinearConv 的 padding 语义不完整

MNIST 两个 `QLinearConv` 使用 `auto_pad=SAME_UPPER`，卷积 kernel 为 `5x5`。ONNX 输出 shape 要求卷积保持空间尺寸，因此 CMSIS-NN 端需要显式 `padding=2`。

修复前逻辑：

- converter 没有完整保留并解释 `auto_pad`。
- codegen 最终可能生成 `padding.h = 0`、`padding.w = 0`。
- 生成代码声明的输出尺寸与实际卷积参数不匹配，后续层输入被污染。

修复后逻辑：

- converter 保留 `Conv/QLinearConv` 的 `auto_pad` 属性。
- `_effective_conv_pads()` 根据 input/output shape、kernel、stride、dilation 推导 `SAME_UPPER/SAME_LOWER/VALID` 的显式 pads。
- `QLinearConv` 的 CMSIS-NN 参数使用推导后的 padding。

该修复让结构验收中的卷积参数正确，但 MNIST 端到端输出仍然存在饱和，说明问题不止在 padding。

### 3. 缺少 ONNX-vs-C 数值测试入口

早期测试只检查“能生成、能编译、能运行”，无法发现分类结果错误。人工 demo 暴露出：ONNX Runtime 能给出有效分类，但生成 C 输出几乎全部饱和，top1 不一致。

由此重构 TDD 目录：

- `tdd/fixtures/` 保存真实 ONNX、数据集和人工采集样本。
- `tdd/work/` 作为 runner 唯一可清理目录，并用 `.nanoc_tdd_workdir` marker 防误删。
- `tdd/results/` 只存报告，不存人工资产。
- `run_numeric_tests.py` 引入 ONNX Runtime 与生成 C 的并行推理对比。

这一步的意义是把 MNIST 问题从临时 demo 提升为可重复的 TDD 失败。

### 4. CNN 输出进入 FC 前布局错位

ONNX 图语义是 NCHW。CMSIS-NN 卷积/池化为了内核效率，运行期中间 buffer 按 NHWC 内存顺序保存。

MNIST 在第二个池化后通过 `Reshape` 进入 FC。ONNX 的 `Reshape/Flatten` 应按 NCHW 逻辑展平；如果 codegen 直接把 CMSIS-NN 的 NHWC 内存交给 FC，特征顺序会错位。

修复前逻辑：

- `Reshape/Flatten` 被当作 alias。
- FC 直接读取上游池化输出 buffer。
- 代码表面上能运行，但类别特征排列与 ONNX 不一致。

修复后逻辑：

- codegen 检测 `4D -> 1D/2D` 的 `Reshape/Flatten` alias 链。
- 为需要展平的张量生成静态中间 buffer。
- 在 FC 前显式执行 NHWC memory -> ONNX NCHW flatten order 的重排：

```text
dest[n*C*H*W + c*H*W + h*W + w]
  = src[n*H*W*C + h*W*C + w*C + c]
```

这个修复把“能连接层”推进到“按 ONNX 语义连接层”。

### 5. QLinearMatMul 错误压扁 per-channel scale

MNIST 的 FC 权重带 per-channel scale。早期为了适配 per-tensor `arm_fully_connected_s8`，converter 将每个输出通道的权重 scale 压到一个 common scale，并重标定权重值。

修复前逻辑：

- 权重从 ONNX 原始 int8 值被重标定到 common scale。
- 每个输出类别的独立量化比例被破坏。
- codegen 调用 `arm_fully_connected_s8`，只能使用 per-tensor quant params。

修复后逻辑：

- converter 保留 QLinearMatMul 原始 int8 权重值，只做 CMSIS-NN FC 需要的转置。
- converter 为每个输出通道生成独立 `multiplier/shift`。
- 中间表示记录 `api: arm_fully_connected_per_channel_s8`。
- codegen 为 FC multiplier/shift 生成数组，并调用 `arm_fully_connected_per_channel_s8`。

这一步将 FC 从“可运行的近似映射”改为“保持 ONNX per-channel 量化语义的映射”。

### 6. QLinearAdd 固定 left_shift 导致饱和

MNIST 的 FC 后存在 `QLinearAdd`，用于把 MatMul 输出和 bias/偏置路径相加。ONNX `QLinearAdd` 的核心语义是按输入 scale 与输出 scale 做重标定后相加。

修复前逻辑：

- converter 固定生成 `left_shift=20`。
- multiplier/shift 未按该放大量做成套抵消。
- Add 阶段把结果推向 `-128/127`，形成高比例饱和。

修复后逻辑：

- `QLinearAdd` 使用 `left_shift=0`。
- 两路输入仍按各自 `input_scale/output_scale` 生成 multiplier/shift。
- 输出饱和比例从接近全饱和降为 `0.00`。

该修复是 MNIST 数值闭环的关键点之一，也提醒后续所有 elementwise 量化算子都必须按 CMSIS-NN API 的 shift 语义审查。

### 7. 数据集与采集 UI 固化

为了避免人工 demo 再次被误删，MNIST 数据集被提升为永久 fixture：

- `tdd/fixtures/datasets/mnist_synthetic_smoke/`
- `tdd/fixtures/datasets/mnist_hand_drawn/`

同时新增本地采集 UI：

```bash
python tdd/tools/mnist_capture/server.py
```

采集的 28x28 灰度样本写入 `tdd/fixtures/datasets/mnist_hand_drawn/dataset.json`，可直接进入 ONNX-vs-C 数值测试。

## 生成器能力前后对比

| 维度 | 修复前 | 修复后 |
|------|--------|--------|
| 成功判定 | 倾向看 `status=ok` 和 smoke run | 以生成物真实 API、编译运行、ONNX-vs-C 数值一致共同判定 |
| Conv padding | `auto_pad` 语义可能丢失，SAME 卷积生成 padding=0 | converter 推导显式 pads，CMSIS-NN 参数与 ONNX shape 对齐 |
| 中间布局 | Conv/Pool 的 NHWC buffer 直接 alias 给 FC | 检测 4D flatten alias，FC 前显式恢复 NCHW flatten 顺序 |
| FC 量化 | per-channel 权重量化被压到 common scale | 保留原始 int8 权重和 per-channel multiplier/shift |
| FC CMSIS-NN API | `arm_fully_connected_s8` | `arm_fully_connected_per_channel_s8` |
| Add 量化 | 固定 `left_shift=20`，输出易饱和 | `left_shift=0`，按输入/输出 scale 重标定 |
| 测试资产 | 人工 demo 放在可能清理的 tmp/results 下 | 数据集进入 `fixtures/`，runner 只清理 `work/` |
| 数值测试 | 无稳定 ONNX-vs-C 对比入口 | `run_numeric_tests.py` 支持 dataset override 和 label accuracy |

## 当前验证结果

结构验收：

```text
run_tests.py --mode target --generate
17/17 PASS
```

合成 smoke 数据集：

```text
dataset=mnist_synthetic_smoke
PASS
top1=10/10
saturation_ratio=0.00
max_abs_error=0.0
```

手写采集数据集：

```text
dataset=mnist_hand_drawn
PASS
sample_count=10
top1=10/10
onnx_label_accuracy=0.70
c_label_accuracy=0.70
label_accuracy_delta=0.0
saturation_ratio=0.00
max_abs_error=59.9307861328125
```

其中 `label_accuracy=0.70` 不是 C 生成失败，而是原始 ONNX 与生成 C 对人工标签的预测结果相同；部分样本标签被故意写错或书写形态与模型判断不一致。该用例真正证明的是：生成 C 对原始 ONNX 的 top1 行为保持一致。

## 已沉淀的工程规则

- `status=ok` 不是最终成功标准，生成物和数值结果才是最终证据。
- 完整网络必须进入数值验收；单算子和结构 smoke 不能代替端到端推理对比。
- 人工数据集必须放入 `tdd/fixtures/`，不得放入 `tdd/work/` 或 runner 可清理目录。
- codegen 不直接解析 ONNX；若生成阶段缺字段，应升级 converter schema。
- 张量布局转换必须显式生成代码或显式记录，不能依赖 NCHW/NHWC 互通假设。
- 量化算子的 multiplier/shift 必须按 CMSIS-NN API 语义审查，尤其是 left shift、rounding 和 saturation。

## 潜在隐患

### 1. flatten 修复目前覆盖的是已知 4D -> 1D/2D 场景

当前逻辑主要服务 MNIST 这类 `NCHW Conv/Pool -> Reshape/Flatten -> FC` 链路。若后续模型出现多 batch、非 4D 输入、动态 shape、显式 `Transpose`、复杂 view 链，仍需要扩展布局分析。

### 2. `QLinearAdd left_shift=0` 需要更多模型验证

MNIST 场景下该修复解决了饱和问题，但 CMSIS-NN elementwise add 的最佳参数组合可能因输入 scale 关系不同而变化。后续应增加更多 QLinearAdd 数值用例，覆盖相近 scale、悬殊 scale、广播 bias 和大幅值输入。

### 3. per-channel FC 依赖 CMSIS-NN API 版本

`arm_fully_connected_per_channel_s8` 需要目标 CMSIS-NN 版本提供并保持签名兼容。生成报告应继续记录 CMSIS-NN 版本约束，固件集成时不能默认所有工程已有该 API。

### 4. 当前数值验收以 host 端 CMSIS-NN 为主

host 编译运行能证明生成 C 逻辑和 CMSIS-NN 调用参数正确，但还不能完全覆盖 Cortex-M 编译器、对齐、栈/静态内存和目标板性能问题。后续需要引入 Arm GNU Toolchain 或目标板/FVP 验证。

### 5. max_abs_error 仍需解释边界

手写数据集中出现非零 `max_abs_error`，但 top1 完全一致且无饱和。后续如果把验收从 top1 一致推进到逐 logit 数值阈值，需要明确 ONNX float 输出、C int8 输出反量化、rounding 差异和输出 scale 的比较规则。

### 6. MNIST 通过不等于任意 int8 ONNX 通过

MNIST 覆盖了一个典型低算力图像分类链路，但仍不代表任意 int8 ONNX。当前结论应限定为：该项目已支持并验证一类直接 QLinear MNIST CNN，包含 QLinearConv、MaxPool、Reshape/Flatten、QLinearMatMul、QLinearAdd 和 DequantizeLinear 的端到端生成与数值一致。

## 后续建议

- 将 `TOPO_003` 的数值测试纳入常规回归，防止布局和量化修复退化。
- 增加逐层 dump 工具，支持在 ONNX 节点输出与 C 中间 buffer 之间定位误差来源。
- 为 `QLinearAdd`、`Flatten/Reshape`、per-channel FC 分别补独立最小数值用例，避免完整网络失败时定位成本过高。
- 增加真实小型模型族，例如 keyword spotting、tiny anomaly detection、简单 IMU 分类，继续扩大低算力嵌入式场景覆盖。
