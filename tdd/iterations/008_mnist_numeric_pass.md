# 迭代 008: MNIST 生成 C 数值推理打通

## 基本信息

- **日期**: 2026-07-04
- **前置迭代**: 007
- **触发来源**: `TOPO_003` 数值验收失败，生成 C 输出饱和且 top1 与 ONNX 不一致

## 目标

让真实 MNIST QLinear int8 ONNX 经过 NanoC-NN 生成的 CMSIS-NN C 代码后，能够在 host 端链接 CMSIS-NN 运行，并与原始 ONNX Runtime 输出保持一致。

## 失败现象

迭代 007 后，结构测试已经通过，但数值测试失败：

```text
top1=1/3
saturation_ratio=0.97
```

修复 `auto_pad=SAME_UPPER` 后，卷积 padding 正确，但 C 输出仍然接近全饱和。

## 根因与修复

### 1. CNN 输出进入 FC 前缺少布局恢复

CMSIS-NN 的 Conv/Pool 中间 buffer 按 NHWC 内存布局保存；ONNX 的 `Reshape`/`Flatten` 语义按 NCHW 展平。

修复：

- codegen 检测 `4D -> 2D` 的 `Reshape/Flatten` alias 链。
- 在 FC 前生成一个静态中间 buffer。
- 将 NHWC 内存重排为 ONNX NCHW flatten 顺序后再调用 FC。

### 2. QLinearMatMul 错误压扁 per-channel weight scale

MNIST FC 权重是 per-channel scale。此前 codegen 为了调用 per-tensor `arm_fully_connected_s8`，将权重缩放到 common scale，会破坏每个类别的量化比例。

修复：

- converter 保留 QLinearMatMul 原始 int8 权重。
- 为每个输出通道生成独立 multiplier/shift。
- codegen 生成 `arm_fully_connected_per_channel_s8` 调用。

### 3. QLinearAdd left_shift 固定为 20 导致输出饱和

ONNX `QLinearAdd` 语义是按输入 scale 与输出 scale 直接重标定后相加。当前 MNIST 的 Add 主要是 MatMul 输出加一个很小 scale 的 bias。

此前固定 `left_shift=20`，但 multiplier 未抵消这个放大，导致 Add 阶段输出被推到 `-128/127`。

修复：

- `QLinearAdd` 使用 `left_shift=0`。
- 输入和输出仍按各自 `scale/output_scale` 生成 multiplier/shift。

## 数据集扩展

`mnist_synthetic_smoke` 从 3 个样本扩展到 10 个固定 pattern，用于更稳定地验证 ONNX-vs-C 并行推理一致性。

该数据集不宣称真实 MNIST 准确率，只用于验证生成 C 与原始 ONNX 的数值一致性。

## 验证结果

数值验收：

```bash
python tdd/scripts/run_numeric_tests.py --case TOPO_003 --generate
```

结果：

```text
PASS
top1=10/10
saturation_ratio=0.00
max_abs_error=0.0
```

结构验收：

```bash
python tdd/scripts/run_tests.py --mode target --generate
```

结果：

```text
17/17 PASS
```

## 当前结论

`TOPO_003` 已经从“结构可生成”推进到“ONNX-vs-C 数值一致”。这代表 MNIST QLinear int8 完整网络在当前 host CMSIS-NN 验收路径上已经打通。

下一步应引入真实手写样本或标准 MNIST 小批量 fixture，将“ONNX-vs-C 一致性”继续推进到“与标签准确率差异可接受”。
