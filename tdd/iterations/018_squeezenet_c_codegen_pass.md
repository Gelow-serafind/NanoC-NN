# 018: SqueezeNet 完整 C 结构生成通过

## 背景

本轮目标是遵循 ONNX 支持图谱和 TDD 开发准则，继续迭代 `NET_001` SqueezeNet 1.0 int8，直到它能够生成可编译、可运行、包含真实 CMSIS-NN 调用路径的 C 推理工程。

上一轮 `AVGPOOL_001` 已经从 SqueezeNet 尾部提炼出 Microsoft 扩展 `QLinearGlobalAveragePool` 最小用例，但完整网络仍然不能晋升为 `ok`。因此本轮继续按真实模型失败点反向提炼最小 case，而不是直接对完整网络硬修。

## 图谱驱动的拆解

`NET_001` 的 fire module 和尾部分类头暴露出三个关键缺口：

1. `DequantizeLinear -> Concat -> QuantizeLinear` 中不同量化参数分支的拼接。
2. `Concat -> MaxPool -> QuantizeLinear` 边界处，池化输入张量缺少可直接用于 CMSIS-NN 的量化信息。
3. `DequantizeLinear -> Softmax -> float output` 的终端输出策略。

对应沉淀为：

- `MAXPOOL_002`: SqueezeNet-style `DQ -> MaxPool -> Q` 边界。
- `CONCAT_002`: 不同 scale 分支 concat，C 侧先 requantize 再调用 `arm_concatenation_s8_z`。
- `SOFTMAX_002`: float output softmax 边界，C 侧以 int8 softmax 概率输出表达。

这些 case 都被登记到 case registry 和 support map，成为后续能力保护的一部分。

## 生成器修复

### Concat 输入重量化

旧逻辑要求 Concat 输入和输出量化参数一致，遇到 SqueezeNet fire module 时会 blocked。

新逻辑在 converter 中保留每个 Concat 输入到输出量化域的 requantize 参数；codegen 为需要转换的输入分配临时 buffer，逐元素调用 CMSIS-NN support function `arm_nn_requantize`，再进入 `arm_concatenation_s8_z`。

这样 `CONCAT_002` 能同时完成结构验收和 ONNX-vs-C 数值验收。

### Pool 边界量化传播

SqueezeNet 中部分 MaxPool 输入来自 `Concat` 的 float-domain 输出，再由后继 `QuantizeLinear` 给出量化参数。旧逻辑在池化节点处无法得到输入量化信息。

新逻辑在 converter 中为 `MaxPool`、`AveragePool`、`GlobalAveragePool` 增加边界量化反向传播：当池化输出已有量化参数而输入缺失时，将输出量化参数用于池化输入。这匹配池化不改变数值域的 CMSIS-NN 使用方式。

该能力由 `MAXPOOL_002` 独立保护。

### Softmax 末端输出策略

SqueezeNet 末端是 `DequantizeLinear -> Softmax -> float output`。生成 C 工程时不能把最终路径变成空 stub。

新逻辑在 converter 中为缺失量化信息的 Softmax 输出推断 CMSIS-NN 兼容的 int8 softmax 输出量化参数：scale `1/256`，zero_point `-128`。codegen 继续映射到 `arm_softmax_s8`。

该能力由 `SOFTMAX_002` 独立保护，并已完成数值 smoke。

## 验证结果

单项用例：

```text
MAXPOOL_002  expected=ok actual=ok api=OK cc=OK run=OK
CONCAT_002   expected=ok actual=ok api=OK cc=OK run=OK
SOFTMAX_002  expected=ok actual=ok api=OK cc=OK run=OK
NET_001      expected=ok actual=ok api=OK cc=OK run=OK
```

数值用例：

```text
CONCAT_002   top1=2/2 max_abs=0.2 cc=OK run=OK
SOFTMAX_002  top1=2/2 max_abs=0.0 cc=OK run=OK
```

全量 target：

```text
python tdd/scripts/run_tests.py --mode target --generate
33/33 PASS
```

稳定回归：

```text
python tdd/scripts/run_regression.py --generate
baseline structural 15/15 PASS
numeric 10/10 PASS
```

## 对 SqueezeNet 的结论

`NET_001` 当前已经从 `blocked` 晋升为结构 `ok`：

- 可以导入 `tdd/fixtures/onnx/squeezenet1.0-12-int8.onnx`。
- 可以解析完整网络结构和量化参数。
- 可以生成包含真实 CMSIS-NN 调用的 C 工程。
- `nanoc_model_run()` 不再是 fallback stub 或空路径。
- 生成物通过 C99 smoke compile/run。

这代表 SqueezeNet 已经达到“完整 C 结构生成”验收，不代表 ImageNet 数据集上的 ONNX-vs-C 数值一致性已经完成。

## 当前边界

已确认：

- `QLinearConv` fire module 主体。
- `MaxPool` SqueezeNet 边界。
- 不同量化分支 `Concat`。
- `QLinearGlobalAveragePool`。
- 末端 `Softmax` CMSIS-NN int8 输出。
- 完整 `NET_001` 结构生成、C99 编译和 host smoke run。

尚未确认：

- SqueezeNet ImageNet 风格样本的 ONNX-vs-C 数值一致性。
- 真实 Cortex-M3/M4/M7 SRAM/Flash 预算下的 `oversize` 行为。
- 完整网络逐层误差定位工具。
- `QLinearGlobalAveragePool` 的独立 ONNX-vs-C 数值验收。

## 对新范式的评价

这轮验证了新的 schema/support-map + TDD 范式是有效的：

- 真实网络先暴露问题。
- 问题回到图谱，判断属于官方 ONNX 算子、扩展算子还是已支持形态的特殊分支。
- 每个缺口先变成最小 case。
- 最小 case 驱动 converter/codegen 修复。
- 完整网络再晋升为能力集。

相比直接修完整网络，这种方式让能力增长有来源、有记录，也能避免 SqueezeNet 通过后下一次类似模型又退化。
