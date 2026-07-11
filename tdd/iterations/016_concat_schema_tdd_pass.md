# 016: Concat schema-driven TDD pass

## 背景

本轮目标是验证新的 ONNX schema 表驱动 TDD 范式是否能指导一个具体算子从 planned/blocked 进入可声明能力集。选择的算子是 ONNX 官方 `Concat`，来源于 `NET_001` SqueezeNet 暴露出的多分支 concat 缺口。

本轮绑定的官方基线：

- ONNX package: `onnx==1.22.0`
- 官方 schema: `ai.onnx.Concat`
- 当前确认子形态: rank=4 NCHW、`axis=1` channel concat、所有输入输出同量化 QDQ/int8
- CMSIS-NN lowering: `arm_concatenation_s8_z`

## 新增测试资产

- `tdd/cases/core/concat/CONCAT_001.md`
- `tdd/fixtures/datasets/concat_channel_smoke/`
- `tdd/scripts/generate_models.py::gen_concat_001`
- `tdd/scripts/cases_registry.py::CONCAT_001`
- `tdd/scripts/support_matrix.py::ONNX_CONCAT_QDQ_INT8`

`CONCAT_001` 使用两个 `[1,1,2,2]` 输入，在 channel 维拼接为 `[1,2,2,2]`。数值验收比较 ONNX Runtime 与生成 C 的输出，覆盖正值、负值和零值。

## 修复内容

### 1. Concat 量化 contract

converter 新增 `Concat` 的同量化约束提取。对于 byte-copy 型 concat，要求所有输入和输出的 `scale`、`zero_point`、`zero_point_dtype` 一致；满足时写入 `model_graph.json.quantization.nodes`，使 int8 contract 可以正确判定为 `ok`。

### 2. shape-helper Concat 过滤

全量回归暴露 `NET_002` MobileNetV2 中的 `Concat_102` 是 shape construction helper，而不是运行时 tensor concat。mapper 已经将其 folded，但 converter contract 初版误将其纳入 runtime 节点，导致 `NET_002` 从 `ok` 回归为 `blocked`。

本轮在 converter contract 中增加 shape-helper Concat 过滤：输入输出 shape 均为 rank<=1 的 Concat 不参与 int8 runtime contract。

### 3. 多输入 C ABI 打包

`nanoc_model_run(const int8_t *input, int8_t *output)` 的 `NANOC_MODEL_INPUT_BYTES` 已经按所有 graph input 求和，但 codegen 之前把所有 model input 都映射到同一个 `input` 指针。Concat 最小数值用例需要两个输入，因此本轮修复为按 `model_graph.json.inputs` 顺序映射：

- 第一个输入: `input`
- 第二个输入: `input + first_input_size`
- 后续输入依次累加 offset

numeric runner 同步支持多输入样本，将 dataset 中的 `inputs` 按 graph input 顺序量化并打包到同一段 input buffer。

### 4. 输出转置缓冲

当 CMSIS-NN 内部 NHWC 输出需要转回 ONNX NCHW 时，若最后一层直接写入 `output`，原逻辑存在原地转置覆盖风险。本轮增加 `nanoc_output_nhwc` 中间 buffer，让最后一层先写 NHWC，再拷贝转置到外部 `output`。

## 验证结果

结构回归：

```text
python tdd/scripts/run_tests.py --mode target --generate
29/29 PASS
```

数值回归：

```text
python tdd/scripts/run_numeric_tests.py --generate
8/8 PASS
```

Concat 单项数值结果：

```text
CONCAT_001 top1=2/2 sat=0.00 max_abs=0.0 cc=OK run=OK
```

## 新范式评价

这轮迭代证明新的表驱动 TDD 范式是有效的：

- support matrix 先定义了官方 schema 子形态，避免“泛泛支持 Concat”的过度声明。
- case registry 强制 `CONCAT_001` 绑定 `ONNX_CONCAT_QDQ_INT8`，validate 阶段能检查用例没有脱离 ONNX 能力地图。
- 首次全量回归暴露了 `NET_002` 的 shape-helper Concat 误伤，说明 schema 范式不仅能扩能力，也能约束“运行时算子”和“图构造 helper”的边界。
- 能力声明最终由 `CAPABILITIES.md` 自动生成，Concat 只有在结构和数值测试通过后才进入产品能力集。

## 当前边界

已确认支持：

- ONNX `Concat`
- rank=4 NCHW
- `axis=1` channel concat
- QDQ/int8 同量化输入输出
- CMSIS-NN `arm_concatenation_s8_z`
- 多 graph input 按 input buffer offset 打包

尚未确认：

- `axis=0/2/3` 的 batch/height/width concat
- 不同量化参数下的 concat/requantize 策略
- 动态 shape concat
- NHWC 原生模型输入的 concat
- 三个及以上运行时输入 concat
- 完整 `NET_001` SqueezeNet 中更复杂的 Concat/GlobalAveragePool/Softmax 边界
